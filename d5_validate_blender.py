"""Run in an isolated background Blender; imports never touch an open session."""
import bpy
import json
import sys
from pathlib import Path
from mathutils import Vector

root = Path(sys.argv[sys.argv.index("--") + 1])
asset_name = sys.argv[sys.argv.index("--") + 2]
inventory = json.loads((root / "资产清单.json").read_text("utf-8"))
asset = next(a for a in inventory["assets"] if a["name"] == asset_name)
bpy.ops.wm.read_factory_settings(use_empty=True)
print("IMPORT_START", asset["gltf"], flush=True)
kwargs = {"filepath": str(root / asset["gltf"])}
props = bpy.ops.import_scene.gltf.get_rna_type().properties.keys()
if "merge_vertices" in props:
    kwargs["merge_vertices"] = False
bpy.ops.import_scene.gltf(**kwargs)
objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
counts = dict(mesh_objects=len(objects), vertices=sum(len(o.data.vertices) for o in objects),
              triangles=sum(len(o.data.polygons) for o in objects))
assert counts["mesh_objects"] == asset["exported_groups"], counts
assert counts["triangles"] == asset["expected_blender_triangles"], counts
by_name = {g["name"]: g for g in asset["groups"] if "individual_gltf" in g}
for obj in objects:
    assert len(obj.data.polygons) == by_name[obj.name]["expected_blender_triangles"], obj.name
images = [dict(name=im.name, filepath=im.filepath, size=list(im.size), loaded=bool(im.has_data)) for im in bpy.data.images if im.source == "FILE"]
assert all(im["loaded"] and all(im["size"]) for im in images), images
report = dict(blender_version=bpy.app.version_string, asset=asset_name, status="passed", counts=counts,
              gltf_triangles=asset["triangles"], removed_duplicate_or_repeated_index_faces=asset["triangles"] - counts["triangles"],
              images=images, objects=[dict(name=o.name, triangles=len(o.data.polygons), vertices=len(o.data.vertices)) for o in objects])
report_path = root / "验证与说明" / f"{asset_name}_Blender导入验证.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("IMPORT_VALIDATED", json.dumps(counts), flush=True)

if asset_name == "11":
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    # Inspection view of warehouse internals; only preview hides the enclosure.
    # The exported glTF retains every original mesh including walls and ground.
    for obj in objects:
        if obj.name in ("外墙", "wall", "17 - Default"):
            obj.hide_render = True
    cam = bpy.data.cameras.new("Inspection Camera")
    camera = bpy.data.objects.new("Inspection Camera", cam)
    scene.collection.objects.link(camera)
    camera.location = (-213, -138, 140)
    target = Vector((-76, 8, 12))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    cam.type = "ORTHO"
    cam.ortho_scale = 175
    cam.clip_end = 3000
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    shade = scene.display.shading
    shade.light = "STUDIO"
    shade.color_type = "MATERIAL"
    shade.show_shadows = True
    shade.show_cavity = True
    shade.cavity_type = "WORLD"
    shade.show_object_outline = False
    shade.background_type = "WORLD"
    scene.world = bpy.data.worlds.new("Inspection World")
    scene.world.color = (0.15, 0.17, 0.2)
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1100
    scene.render.resolution_percentage = 100
    choices = [i.identifier for i in scene.render.image_settings.bl_rna.properties["file_format"].enum_items]
    assert "PNG" in choices
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(root / "验证与说明" / "11_模型结构预览.png")
    bpy.ops.render.render(write_still=True)
    print("PREVIEW_RENDERED", flush=True)
