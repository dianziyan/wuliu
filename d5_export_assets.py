"""Recover this project's unencrypted D5 mesh versions 10/11 as glTF 2.0.

Original files are read only. NumPy is the only non-standard dependency.
This is a format-specific recovery tool, not an official D5 exporter.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"Truncated mesh at {self.f.tell()}, expected {n} bytes")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def string(self, wide=False):
        n = self.u32()
        if n > 1_000_000:
            raise ValueError(f"Invalid string length {n}")
        return self.read(n * (2 if wide else 1)).decode("utf-16le" if wide else "utf-8").rstrip("\0")

    def array(self, dtype, count=None):
        n = self.u32() if count is None else count
        return np.frombuffer(self.read(n * 4), dtype=dtype)


def parse_mesh(path):
    """Index arrays without loading all mesh data into memory."""
    nodes, meshes = [], []
    with path.open("rb") as f:
        r = Reader(f)
        version = r.u32()
        if version not in (10, 11):
            raise ValueError(f"Unsupported D5 mesh version {version}")
        header = json.loads(r.string(version == 11))
        if version == 11:
            assert r.u32() == 0
            for i in range(r.u32()):
                name, label = r.string(True), r.string(True)
                matrix = np.array(struct.unpack("<16f", r.read(64))).reshape(4, 4)
                # All current nodes are uniform source-inches to D5-centimetres.
                expect = np.diag([matrix[0, 0]] * 3 + [1.0])
                assert np.allclose(matrix, expect), (name, matrix)
                nodes.append(dict(name=name, label=label, material_index=i, scale_cm=float(matrix[0, 0]), matrix=matrix.tolist()))
            for i in range(r.u32()):
                mesh = dict(name=r.string(True), arrays=[])
                for semantic in ("position", "normal", "uv", "extra", "indices"):
                    count = r.u32()
                    mesh["arrays"].append(dict(semantic=semantic, count=count, offset=f.tell()))
                    f.seek(count * 4, 1)
                c = [a["count"] for a in mesh["arrays"]]
                assert c[0] == c[1] and c[0] % 3 == 0 and c[2] == c[0] // 3 * 2 and c[3] == 0 and c[4] % 3 == 0
                mesh.update(vertex_count=c[0] // 3, triangle_count=c[4] // 3)
                meshes.append(mesh)
        else:
            for i in range(r.u32()):
                name, nv = r.string(), r.u32()
                off = f.tell()
                f.seek(nv * 32, 1)
                ni = r.u32()
                io = f.tell()
                f.seek(ni * 4, 1)
                assert ni % 3 == 0
                meshes.append(dict(name=name, vertex_count=nv, triangle_count=ni // 3, vertex_offset=off, index_offset=io))
            for i in range(r.u32()):
                name, label = r.string(), r.string()
                transform = struct.unpack("<9f", r.read(36))
                assert np.allclose(transform[:6], 0) and np.allclose(transform[6:], transform[6])
                nodes.append(dict(name=name, label=label, material_index=i, scale_cm=float(transform[6]), transform=list(transform)))
            assert r.u32() == 0
        assert f.tell() == path.stat().st_size, (f.tell(), path.stat().st_size)
    by_name = {n["name"]: n for n in nodes}
    assert len(by_name) == len(nodes) == len(meshes)
    for mesh in meshes:
        mesh["node"] = by_name[mesh["name"]]
    return dict(version=version, header=header, nodes=nodes, meshes=meshes, bytes=path.stat().st_size)


def load_arrays(path, version, mesh):
    with path.open("rb") as f:
        if version == 11:
            arrays = []
            for a in mesh["arrays"]:
                f.seek(a["offset"])
                arrays.append(np.fromfile(f, dtype="<u4" if a["semantic"] == "indices" else "<f4", count=a["count"]))
            v = np.empty((mesh["vertex_count"], 8), dtype="<f4")
            v[:, :3] = arrays[0].reshape(-1, 3)
            v[:, 3:6] = arrays[1].reshape(-1, 3)
            v[:, 6:8] = arrays[2].reshape(-1, 2)
            indices = arrays[4]
        else:
            f.seek(mesh["vertex_offset"])
            raw = np.fromfile(f, dtype="<f4", count=mesh["vertex_count"] * 8).reshape(-1, 8)
            v = raw[:, [0, 1, 2, 5, 6, 7, 3, 4]].copy(order="C")
            f.seek(mesh["index_offset"])
            indices = np.fromfile(f, dtype="<u4", count=mesh["triangle_count"] * 3)
    assert np.isfinite(v[:, :3]).all() and np.isfinite(v[:, 6:8]).all(), mesh["name"]
    assert not len(indices) or indices.max() < len(v), mesh["name"]
    bad = ~np.isfinite(v[:, 3:6]).all(axis=1)
    mesh["nonfinite_source_normals_repaired"] = int(bad.sum())
    if bad.any():
        # Recover invalid source normals from incident faces. Source winding is
        # clockwise, so negate the conventional right-handed cross product.
        faces = indices.reshape(-1, 3)
        relevant = faces[bad[faces].any(axis=1)]
        p = v[relevant, :3].astype(np.float64)
        cross = -np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for vertex in np.flatnonzero(bad):
            normal = cross[(relevant == vertex).any(axis=1)].sum(axis=0)
            length = np.linalg.norm(normal)
            v[vertex, 3:6] = normal / length if length > 1e-12 else (0, 0, 1)
    return v, indices


def scalar(params, key, default):
    value = str(params.get(key, ""))
    m = re.search(r"X=([-+\d.eE]+)", value)
    return float(m.group(1)) if m else default


class Exporter:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.textures = {}
        self.warnings = []
        self.inventory = dict(project="lf", model_resources=2, placed_instances=1, assets=[], textures=[], warnings=self.warnings)
        save = json.loads((source / "asset/1.save").read_text("utf-8-sig"))
        offset = save["furnitureDatas"][0]["zeroPointOffset"]
        self.origin_cm = np.array([offset[k] for k in ("x", "y", "z")], dtype=np.float64)
        self.inventory["source_zero_point_offset_cm"] = self.origin_cm.tolist()

    def texture(self, rel, archive=None):
        rel = rel.replace("\\", "/").lstrip("/")
        key = ("archive" if archive else "current", rel)
        if key in self.textures:
            return self.textures[key]
        if archive:
            data = archive.read("textures/" + rel)
        else:
            f = self.source / "asset" / rel
            if not f.is_file():
                self.warnings.append(f"Missing texture: {rel}")
                return None
            data = f.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        dest = self.output / "贴图" / rel
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            dest = self.output / "贴图" / "5_备用资源" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        if not any(t["path"] == dest.relative_to(self.output).as_posix() for t in self.inventory["textures"]):
            self.inventory["textures"].append(dict(path=dest.relative_to(self.output).as_posix(), source=rel, sha256=digest, bytes=len(data)))
        self.textures[key] = dest
        return dest

    def materials(self, info, doc, folder, archive):
        elements = json.loads(info["detailInfo"])["styleDatas"][0]["elements"]
        param_map = {}
        image_map = {}

        def add_texture(rel):
            dest = self.texture(rel, archive)
            if dest is None:
                return None
            if str(dest) not in image_map:
                image_map[str(dest)] = len(doc["images"])
                uri = quote(os.path.relpath(dest, folder).replace("\\", "/"), safe="/.")
                doc["images"].append(dict(uri=uri, name=dest.name))
                doc["textures"].append(dict(source=len(doc["images"]) - 1, sampler=0))
            return dict(index=image_map[str(dest)])

        for e in sorted(elements, key=lambda e: e["materialIndex"]):
            assert e["materialIndex"] == len(doc["materials"])
            m = e["materialData"]
            params = {p["name"]: p["value"] for p in json.loads(m.get("matInfo") or "[]")}
            param_map[e["materialIndex"]] = params
            col = dict(re.findall(r"([RGBA])=([-+\d.eE]+)", str(params.get("Diffuse (Color)", ""))))
            rgb = [min(1, max(0, float(col.get(k, 1 if params.get("Diffuse Map") else .7)))) for k in "RGB"]
            pbr = dict(baseColorFactor=rgb + [1], metallicFactor=min(1, max(0, scalar(params, "Metallic", 0))), roughnessFactor=min(1, max(0, scalar(params, "Roughness Map (opacity)", .5))))
            mat = dict(name=m.get("title") or f"Material_{e['materialIndex']}", pbrMetallicRoughness=pbr, doubleSided=True)
            if params.get("Diffuse Map"):
                tex = add_texture(params["Diffuse Map"])
                if tex is not None:
                    pbr["baseColorTexture"] = tex
            if params.get("Normal Map One"):
                # These two 'normal' files are byte-identical to their grayscale
                # colour maps. Preserve the source file, but do not treat it as
                # an RGB tangent-space normal texture in glTF.
                self.texture(params["Normal Map One"], archive)
                self.warnings.append(f"{folder.name}/{mat['name']}: source normal image is a grayscale colour-map duplicate; omitted from glTF normal slot.")
            mat["extras"] = dict(d5_material_index=e["materialIndex"], d5_shader=m.get("uePath", ""), d5_parameters=params)
            doc["materials"].append(mat)
        return param_map

    def export(self, path, info, slug, source_name, archive=None):
        parsed = parse_mesh(path)
        folder = self.output / "3D资产" / slug
        folder.mkdir(parents=True, exist_ok=True)
        binpath = folder / "geometry.bin"
        gltfpath = folder / f"{source_name}_完整场景.gltf"
        doc = dict(asset=dict(version="2.0", generator="D5 mesh recovery / NumPy"), scene=0,
                   scenes=[dict(name=source_name, nodes=[])], nodes=[], meshes=[], materials=[],
                   images=[], textures=[], samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)],
                   buffers=[dict(uri="geometry.bin", byteLength=0)], bufferViews=[], accessors=[])
        param_map = self.materials(info, doc, folder, archive)
        result = dict(name=source_name, source_format_version=parsed["version"], source_header=parsed["header"],
                      geometry_groups=len(parsed["meshes"]), exported_groups=0, triangles=0, source_vertices=0,
                      exported_vertices=0, gltf=gltfpath.relative_to(self.output).as_posix(), groups=[])
        result["source_triangles"] = sum(m["triangle_count"] for m in parsed["meshes"])
        result["duplicate_face_or_repeated_index_triangles"] = 0
        low, high = np.full(3, np.inf), np.full(3, -np.inf)

        with binpath.open("wb") as out:
            def accessor(a, semantic, bounds=False):
                pad = (-out.tell()) % 4
                out.write(b"\0" * pad)
                a = np.ascontiguousarray(a, dtype="<u4" if semantic == "INDEX" else "<f4")
                view = len(doc["bufferViews"])
                doc["bufferViews"].append(dict(buffer=0, byteOffset=out.tell(), byteLength=a.nbytes, target=34963 if semantic == "INDEX" else 34962))
                a.tofile(out)
                acc = dict(bufferView=view, componentType=5125 if semantic == "INDEX" else 5126,
                           count=len(a), type="SCALAR" if a.ndim == 1 else f"VEC{a.shape[1]}")
                if bounds:
                    acc.update(min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist())
                doc["accessors"].append(acc)
                return len(doc["accessors"]) - 1

            for i, mesh in enumerate(parsed["meshes"]):
                tic = time.monotonic()
                node = mesh["node"]
                rec = dict(name=node["label"], d5_name=mesh["name"], material_index=node["material_index"], source_vertices=mesh["vertex_count"], triangles=mesh["triangle_count"])
                result["groups"].append(rec)
                if not mesh["triangle_count"]:
                    rec["skipped"] = "empty geometry"
                    continue
                print(f"{source_name} [{i+1}/{len(parsed['meshes'])}] {node['label']}: {mesh['triangle_count']:,} triangles", flush=True)
                vertices, indices = load_arrays(path, parsed["version"], mesh)
                rec["nonfinite_source_normals_repaired"] = mesh["nonfinite_source_normals_repaired"]
                if not np.any(np.ptp(vertices[:, :3], axis=0) > 0):
                    rec["skipped"] = "All vertices collapse to one point; no surface area. Source triangles retained in original D5 archive."
                    print(f"  skipped collapsed group ({mesh['triangle_count']} zero-area triangles)", flush=True)
                    continue
                # Exact equality only: no coordinate rounding, no decimation.
                records = vertices.view(np.dtype((np.void, vertices.dtype.itemsize * 8))).ravel()
                _, keep, inverse = np.unique(records, return_index=True, return_inverse=True)
                vertices = vertices[keep]
                indices = inverse[indices].astype("<u4")
                del records, keep, inverse
                pos = vertices[:, :3].astype(np.float64) * (node["scale_cm"] * .01) - self.origin_cm * .01
                # D5 left-handed Z-up -> glTF right-handed Y-up. The reflection
                # also changes source clockwise triangles to glTF CCW.
                positions = pos[:, [0, 2, 1]].astype("<f4")
                normals = vertices[:, [3, 5, 4]].copy()
                lengths = np.linalg.norm(normals, axis=1)
                valid = lengths > 1e-12
                normals[valid] /= lengths[valid, None]
                normals[~valid] = (0, 1, 0)
                rec["zero_normals_replaced"] = int((~valid).sum())
                uv = vertices[:, 6:8].copy()
                params = param_map[node["material_index"]]
                # D5 imported UVs are stored scaled by 0.0254 for its centimetre
                # UV convention. Retain explicit material tiling where present.
                uv[:, 0] *= scalar(params, "Utiling_D", scalar(params, "Utiling", 1))
                uv[:, 1] = 1 - uv[:, 1] * scalar(params, "Vtiling_D", scalar(params, "Vtiling", 1))
                # Check winding against source normals (without changing faces).
                faces = indices.reshape(-1, 3)
                non_repeated = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
                sorted_faces = np.sort(faces[non_repeated], axis=1)
                valid_unique = len(np.unique(sorted_faces.view("V12").ravel()))
                rec["expected_blender_triangles"] = valid_unique
                rec["duplicate_face_or_repeated_index_triangles"] = len(faces) - valid_unique
                result["duplicate_face_or_repeated_index_triangles"] += len(faces) - valid_unique
                del sorted_faces, non_repeated
                sample = faces[::max(1, len(faces) // 10000)]
                dot = (np.cross(positions[sample[:, 1]] - positions[sample[:, 0]], positions[sample[:, 2]] - positions[sample[:, 0]]) * normals[sample].mean(1)).sum(1)
                rec["sample_winding_positive"] = int((dot > 1e-8).sum())
                rec["sample_winding_negative"] = int((dot < -1e-8).sum())
                p = accessor(positions, "POSITION", True)
                n = accessor(normals, "NORMAL")
                t = accessor(uv, "TEXCOORD_0")
                ind = accessor(indices, "INDEX")
                meshidx = len(doc["meshes"])
                doc["meshes"].append(dict(name=node["label"], primitives=[dict(attributes=dict(POSITION=p, NORMAL=n, TEXCOORD_0=t), indices=ind, material=node["material_index"], mode=4)]))
                doc["nodes"].append(dict(name=node["label"], mesh=meshidx, extras=dict(d5_source_name=mesh["name"], source_material_index=node["material_index"])))
                doc["scenes"][0]["nodes"].append(meshidx)
                mn, mx = positions.min(axis=0), positions.max(axis=0)
                low, high = np.minimum(low, mn), np.maximum(high, mx)
                rec.update(exported_vertices=len(positions), bounds_gltf_m=[mn.tolist(), mx.tolist()], gltf_mesh_index=meshidx)
                result["exported_groups"] += 1
                result["triangles"] += mesh["triangle_count"]
                result["source_vertices"] += mesh["vertex_count"]
                result["exported_vertices"] += len(positions)
                print(f"  vertices {mesh['vertex_count']:,} -> {len(positions):,}, {time.monotonic()-tic:.1f}s", flush=True)
                del vertices, indices, pos, positions, normals, lengths, uv, faces, sample
            doc["buffers"][0]["byteLength"] = out.tell()
        result["bounds_gltf_m"] = [low.tolist(), high.tolist()]
        result["geometry_bytes"] = binpath.stat().st_size
        result["source_header_triangle_difference"] = int(parsed["header"]["triangleCount"] - result["source_triangles"])
        result["excluded_collapsed_triangles"] = result["source_triangles"] - result["triangles"]
        result["expected_blender_triangles"] = result["triangles"] - result["duplicate_face_or_repeated_index_triangles"]
        doc["extras"] = dict(units="metres", source_zero_point_offset_cm=self.origin_cm.tolist(),
                             conversion="gltf = (source_xyz * node_scale_cm - source_zero_point_offset_cm)[X,Z,Y] / 100",
                             source_model=source_name, actual_triangles=result["triangles"],
                             material_note="Approximate PBR translation; original D5 shader parameters retained in extras.")
        # Split buffers by group. Full scene and individual glTF files share
        # these small buffers; importing one group need not read the whole scene.
        original_views = copy.deepcopy(doc["bufferViews"])
        doc["buffers"] = []
        for rec in result["groups"]:
            if "gltf_mesh_index" not in rec:
                continue
            idx = rec["gltf_mesh_index"]
            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", rec["name"]).rstrip(" .")[:80]
            file = folder / "网格分组" / f"{rec['material_index']:03d}_{safe}.gltf"
            file.parent.mkdir(parents=True, exist_ok=True)
            part_bin = file.with_suffix(".bin")
            prim = doc["meshes"][idx]["primitives"][0]
            accessor_ids = list(prim["attributes"].values()) + [prim["indices"]]
            view_ids = [doc["accessors"][a]["bufferView"] for a in accessor_ids]
            start = min(original_views[v]["byteOffset"] for v in view_ids)
            end = max(original_views[v]["byteOffset"] + original_views[v]["byteLength"] for v in view_ids)
            with binpath.open("rb") as src, part_bin.open("wb") as dst:
                src.seek(start)
                remaining = end - start
                while remaining:
                    chunk = src.read(min(remaining, 16 * 1024 * 1024))
                    assert chunk
                    dst.write(chunk)
                    remaining -= len(chunk)
            buffer_id = len(doc["buffers"])
            doc["buffers"].append(dict(uri=quote(part_bin.relative_to(folder).as_posix(), safe="/."), byteLength=end-start))
            for vid in view_ids:
                doc["bufferViews"][vid]["buffer"] = buffer_id
                doc["bufferViews"][vid]["byteOffset"] = original_views[vid]["byteOffset"] - start
            single = dict(asset=doc["asset"], scene=0, scenes=[dict(name=rec["name"], nodes=[0])],
                          nodes=[copy.deepcopy(doc["nodes"][idx])], meshes=[copy.deepcopy(doc["meshes"][idx])],
                          buffers=[dict(uri=quote(part_bin.name, safe="."), byteLength=end-start)],
                          materials=[copy.deepcopy(doc["materials"][prim["material"]])],
                          bufferViews=[], accessors=[], extras=dict(doc["extras"]))
            single["extras"]["actual_triangles"] = rec["triangles"]
            single["nodes"][0]["mesh"] = 0
            single_prim = single["meshes"][0]["primitives"][0]
            single_prim["material"] = 0
            mapping = {aid: i for i, aid in enumerate(accessor_ids)}
            single_prim["attributes"] = {k:mapping[v] for k,v in prim["attributes"].items()}
            single_prim["indices"] = mapping[prim["indices"]]
            for aid, vid in zip(accessor_ids, view_ids):
                acc = copy.deepcopy(doc["accessors"][aid])
                acc["bufferView"] = len(single["bufferViews"])
                view = copy.deepcopy(doc["bufferViews"][vid])
                view["buffer"] = 0
                single["accessors"].append(acc)
                single["bufferViews"].append(view)
            tex = single["materials"][0]["pbrMetallicRoughness"].get("baseColorTexture")
            if tex:
                texture = copy.deepcopy(doc["textures"][tex["index"]])
                image = copy.deepcopy(doc["images"][texture["source"]])
                image["uri"] = "../" + image["uri"]
                texture["source"] = 0
                tex["index"] = 0
                single.update(textures=[texture], images=[image], samplers=doc["samplers"])
            json_write(file, single)
            rec["individual_gltf"] = file.relative_to(self.output).as_posix()
        json_write(gltfpath, doc)
        # This is only the exact temporary buffer created in this function.
        binpath.unlink()
        json_write(folder / "原始材质与节点.json", dict(info=info, nodes=parsed["nodes"]))
        self.inventory["assets"].append(result)
        json_write(self.output / "资产清单.json", self.inventory)
        return result

    def run(self, models=("11", "5")):
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = self.output / "资产清单.json"
        if set(models) != {"11", "5"} and manifest.exists():
            self.inventory = json.loads(manifest.read_text("utf-8"))
            self.inventory["assets"] = [a for a in self.inventory["assets"] if a["name"] not in models]
            self.warnings = self.inventory["warnings"]
        current = self.source / "asset/d5a/c1a8de80033164f3c53faec252ef04aa/1.d5mesh"
        info = json.loads((self.source / "asset/d5a/11443B3E530c5369bb5d123dbffe16ff25fa64e2/info.json").read_text("utf-8-sig"))
        if "11" in models:
            self.export(current, info, "11_当前场景", "11")
        package = self.source / "resource/597bb9168e694455b0a488dffd0e671d.d5a"
        # Stage only the named mesh; never extract arbitrary archive paths.
        stage = self.output / "验证与说明" / "resource_5_staging.d5mesh"
        stage.parent.mkdir(parents=True, exist_ok=True)
        try:
            if "5" in models:
                with zipfile.ZipFile(package) as archive:
                    with archive.open("1.d5mesh") as src, stage.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
                    info = json.loads(archive.read("info.json"))
                    self.export(stage, info, "5_备用资源", "5", archive)
        finally:
            # Only this exact, exporter-created scratch file is removed.
            if stage.exists():
                stage.unlink()
        self.inventory["unique_texture_contents"] = len({t["sha256"] for t in self.inventory["textures"]})
        self.inventory["texture_files"] = len(self.inventory["textures"])
        self.inventory["total_exported_groups"] = sum(a["exported_groups"] for a in self.inventory["assets"])
        self.inventory["total_triangles"] = sum(a["triangles"] for a in self.inventory["assets"])
        self.inventory["notes"] = [
            "2 model resources, only 11.skp is placed in the current D5 scene. 5.skp is a separate resource, not a second current scene instance.",
            "Mesh groups are primarily merged by material; they do not represent individual machines or unique semantic assets.",
            "Source component hierarchy, parametric objects and CAD solids are not present in the recovered meshes.",
            "No decimation of valid mesh groups: only exactly equal position/normal/UV vertices are merged. One backup Global group has all vertices collapsed to one point (260 zero-area triangles) and is excluded from usable assets.",
            "Coordinates use the current scene zero-point offset and metres; both recovered models retain the same coordinate frame.",
            "D5 procedural material effects, lighting, animation and post-processing are not reproduced.",
            "Four current-scene material image files are identical in content, leaving seven unique images among ten source texture files.",
        ]
        json_write(self.output / "资产清单.json", self.inventory)
        print("EXPORT COMPLETE", json.dumps({k:self.inventory[k] for k in ("model_resources", "total_exported_groups", "total_triangles", "texture_files", "unique_texture_contents")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", nargs="+", choices=("11", "5"), default=["11", "5"])
    args = ap.parse_args()
    Exporter(args.source.resolve(), args.output.resolve()).run(args.models)
