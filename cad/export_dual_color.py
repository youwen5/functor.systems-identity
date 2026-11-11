"""Export two-material models with PrusaSlicer volume/tool assignments.

Run with the same Python dependencies as export_prints.py. Rebuilds the shared
source geometry and its original monochrome validation before partitioning it.
STLs are aligned part pairs; 3MF files keep the pair together with colors.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import manifold3d as md
import numpy as np
import trimesh

CAD = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prusa-slicer", help="Optional executable for native PrusaSlicer import/export validation")
args = parser.parse_args()
OUT = CAD / "two-color"
OUT.mkdir(exist_ok=True)
g = runpy.run_path(str(CAD / "export_prints.py"))
union, as_trimesh = g["union"], g["as_trimesh"]
REPORT = {"units": "millimetres", "printer": "Prusa XL, two tools, 0.4 mm nozzles",
          "tool_1": "white", "tool_2": "black", "files": {}, "models": {},
          "source_sha256": g["REPORT"]["source_sha256"]}


def checked_stl(name, solid):
    solid = solid.simplify(0.0002)
    assert solid.status() == md.Error.NoError, (name, solid.status())
    mesh = as_trimesh(solid)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="stl")
    mesh = trimesh.load_mesh(path, process=True)
    shells = mesh.split(only_watertight=False)
    assert mesh.is_watertight and mesh.is_winding_consistent, name
    assert np.isfinite(mesh.vertices).all(), name
    assert all(s.is_watertight and s.is_winding_consistent and s.volume > 0 for s in shells), name
    REPORT["files"][name] = {"watertight": True, "consistent_winding": True,
        "all_shells_positive_volume": True, "shells": len(shells),
        "volume_mm3": float(mesh.volume), "triangles": len(mesh.faces),
        "bounds_mm": mesh.bounds.tolist(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print("Validated", name, len(shells), "closed shells", flush=True)
    return mesh


NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ET.register_namespace("", NS)


def tag(parent, name, attrs=None):
    return ET.SubElement(parent, "{" + NS + "}" + name, attrs or {})


def xml_bytes(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_3mf(name, meshes):
    # PrusaSlicer stores multiple volumes in one object mesh, with face ranges
    # identifying volumes in Slic3r_PE_model.config. Keep vertex indices separate
    # at material interfaces, just as its own exporter does.
    title = name.replace("functor-", "Functor ").replace("-", " ").removesuffix(".3mf")
    root = ET.Element("{" + NS + "}model", {"unit": "millimeter",
        "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
        "xmlns:slic3rpe": "http://schemas.slic3r.org/3mf/2017/06"})
    tag(root, "metadata", {"name": "Title"}).text = title
    tag(root, "metadata", {"name": "Application"}).text = "functor.systems CAD exporter"
    tag(root, "metadata", {"name": "slic3rpe:Version3mf"}).text = "1"
    resources = tag(root, "resources")
    materials = tag(resources, "basematerials", {"id": "1"})
    for color in ["White", "Black"]:
        tag(materials, "base", {"name": color, "displaycolor": "#FFFFFFFF" if color == "White" else "#000000FF"})
    obj = tag(resources, "object", {"id": "2", "type": "model", "name": title, "pid": "1", "pindex": "0"})
    mesh_xml = tag(obj, "mesh")
    vertices = tag(mesh_xml, "vertices")
    triangles = tag(mesh_xml, "triangles")
    settings = ET.Element("config")
    settings_obj = ET.SubElement(settings, "object", {"id": "2", "instances_count": "1"})
    ET.SubElement(settings_obj, "metadata", {"type": "object", "key": "name", "value": title})
    ET.SubElement(settings_obj, "metadata", {"type": "object", "key": "extruder", "value": "1"})
    vertex_start = face_start = 0
    for color_index, mesh in enumerate(meshes):
        color = ["White", "Black"][color_index]
        for xyz in mesh.vertices:
            tag(vertices, "vertex", dict(zip("xyz", (format(float(v), ".10g") for v in xyz))))
        for face in mesh.faces:
            attrs = {"v" + str(i + 1): str(int(v) + vertex_start) for i, v in enumerate(face)}
            attrs.update(pid="1", p1=str(color_index))
            tag(triangles, "triangle", attrs)
        volume = ET.SubElement(settings_obj, "volume", {"firstid": str(face_start),
            "lastid": str(face_start + len(mesh.faces) - 1)})
        for key, value in {"name": f"{color} - tool {color_index + 1}",
                           "volume_type": "ModelPart", "extruder": str(color_index + 1),
                           "matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"}.items():
            ET.SubElement(volume, "metadata", {"type": "volume", "key": key, "value": value})
        vertex_start += len(mesh.vertices)
        face_start += len(mesh.faces)
    build = tag(root, "build")
    # Put the complete assembly at the center of the XL's bed. Parts retain a
    # common local coordinate frame; never independently drop the white volume.
    tag(build, "item", {"objectid": "2", "transform": "1 0 0 0 1 0 0 0 1 180 180 0"})
    content_types = b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
<Default Extension="config" ContentType="application/xml"/>
</Types>'''
    relationships = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rel0" Target="/3D/3dmodel.model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", relationships)
        z.writestr("3D/3dmodel.model", xml_bytes(root))
        z.writestr("Metadata/Slic3r_PE_model.config", xml_bytes(settings))
    validate_3mf(path, meshes)
    return path


def validate_3mf(path, originals):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        root = ET.fromstring(z.read("3D/3dmodel.model"))
        settings = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
    ns = {"m": NS}
    assert root.attrib["unit"] == "millimeter"
    mesh_xml = root.find("m:resources/m:object/m:mesh", ns)
    vertices = np.array([[float(v.attrib[k]) for k in "xyz"] for v in mesh_xml.find("m:vertices", ns)])
    triangles = list(mesh_xml.find("m:triangles", ns))
    faces = np.array([[int(t.attrib[k]) for k in ["v1", "v2", "v3"]] for t in triangles])
    volumes = settings.findall("object/volume")
    assert len(volumes) == 2
    first_expected = 0
    for i, (volume, original) in enumerate(zip(volumes, originals)):
        first, last = int(volume.attrib["firstid"]), int(volume.attrib["lastid"])
        assert first == first_expected
        first_expected = last + 1
        assert volume.find("metadata[@key='extruder']").attrib["value"] == str(i + 1)
        assert all(t.attrib["p1"] == str(i) for t in triangles[first:last + 1])
        part = trimesh.Trimesh(vertices, faces[first:last + 1], process=True)
        part.remove_unreferenced_vertices()
        assert part.is_watertight and part.is_winding_consistent
        assert np.allclose(part.bounds, original.bounds, atol=0.00001)
        assert abs(part.volume - original.volume) < 0.01
    assert first_expected == len(faces)
    REPORT["files"][str(path.relative_to(OUT))] = {"format": "3MF", "tool_assignments": [1, 2],
        "display_colors": ["#FFFFFF", "#000000"], "volume_round_trip_validated": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export_model(name, white, black, complete):
    print("Partitioning", name, flush=True)
    overlap = (white ^ black).volume()
    combined = union([white, black])
    missing = (complete - combined).volume()
    extra = (combined - complete).volume()
    assert max(abs(overlap), abs(missing), abs(extra)) < 0.001, (name, overlap, missing, extra)
    white_mesh = checked_stl(f"stl-parts/{name}-WHITE-tool1.stl", white)
    black_mesh = checked_stl(f"stl-parts/{name}-BLACK-tool2.stl", black)
    bounds = np.array([np.minimum(white_mesh.bounds[0], black_mesh.bounds[0]),
                       np.maximum(white_mesh.bounds[1], black_mesh.bounds[1])])
    size = bounds[1] - bounds[0]
    assert np.all(size < [360, 360, 360]) and abs(bounds[0, 2]) < 0.0001
    assert abs(white_mesh.volume + black_mesh.volume - complete.volume()) / complete.volume() < 0.00001
    write_3mf(f"{name}.3mf", [white_mesh, black_mesh])
    REPORT["models"][name] = {"size_mm": size.tolist(), "fits_prusa_xl": True,
        "intermaterial_overlap_mm3": overlap, "missing_geometry_mm3": missing,
        "extra_geometry_mm3": extra, "same_coordinate_frame": True}


art = union(g["relief"])
plaque_white = g["panel"] - art
plaque_black = union([g["foot_blank"] - g["panel"], art])
export_model("functor-plaque-black-white", plaque_white, plaque_black, g["complete_plaque"])

black = union([g["base"], g["grid"], *g["supports"], *g["arrows"], g["rim"],
               g["frame"], g["letters"], *g["posts"]])
white = union([g["cap"], g["torus_body"], g["ball"], g["face"]]) - black
export_model("functor-sculpture-black-white", white, black, g["sculpture"])


def flatten(solid):
    return g["transformed"](solid, (0, 0, 3), (1, 0, 0), (0, 0, -1), (0, 1, 0)).translate((0, -6, 0))


export_model("optional-flat-plaque/01-panel-black-white", flatten(plaque_white), flatten(art), g["flat_panel"])
checked_stl("optional-flat-plaque/02-stand-BLACK-tool2.stl", g["stand"])


def validate_in_prusaslicer(executable):
    version = subprocess.run([executable, "--help"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    results = {}
    with tempfile.TemporaryDirectory(prefix="functor-prusa-check-") as directory:
        for source in sorted(OUT.rglob("*.3mf")):
            destination = Path(directory) / "roundtrip.3mf"
            result = subprocess.run([executable, "--datadir", str(Path(directory) / "settings"),
                "--info", "--export-3mf", "--output", str(destination), str(source)],
                check=True, capture_output=True, text=True)
            assert "manifold = yes" in result.stdout, result.stdout
            with zipfile.ZipFile(destination) as z:
                settings = ET.fromstring(z.read("Metadata/Slic3r_PE_model.config"))
            objects = settings.findall("object")
            assert len(objects) == 1
            volumes = objects[0].findall("volume")
            assert len(volumes) == 2
            assert [v.find("metadata[@key='extruder']").attrib["value"] for v in volumes] == ["1", "2"]
            assert all(all(int(value) == 0 for value in v.find("mesh").attrib.values()) for v in volumes)
            results[str(source.relative_to(OUT))] = {"manifold": True, "objects": 1,
                "material_volumes": 2, "tool_assignments_preserved": [1, 2], "mesh_repairs": 0}
            print("PrusaSlicer verified", source.name, flush=True)
    REPORT["prusaslicer_validation"] = {"version": version, "files": results,
        "scope": "Native import/export round-trip; no filament-specific slicing or physical print test"}


if args.prusa_slicer:
    validate_in_prusaslicer(args.prusa_slicer)

guide = '''FUNCTOR.SYSTEMS — BLACK / WHITE, PRUSA XL

Two tools, 0.4 mm nozzles. Units: millimetres; use 100% scale.
  Tool 1: WHITE — light backing, torus core, ball, pedestal cap, sign face.
  Tool 2: BLACK — all formerly colored/dark artwork, arrows, wireframe,
                  lettering, structural supports, sign frame and base.
Use the same filament type for both colors so they bond and share settings.

MAIN FILES
  functor-plaque-black-white.3mf: 168 x 56 x 174 mm, assembled upright.
  functor-sculpture-black-white.3mf: 218 x 144 x 202 mm, assembled upright.
Both fit the XL's 360 x 360 x 360 mm build volume at full scale.
Print separately, allowing room for generated supports and a wipe tower.

PRUSASLICER
  Select your Original Prusa XL 2-tool, 0.4 mm printer preset and your actual
  filament presets. Import a 3MF. It contains one object with two material
  volumes, named White - tool 1 and Black - tool 2, assigned to those tools.
  Set the filament display colors to white and black in your chosen presets.
  The file includes standard 3MF display colors and PrusaSlicer tool metadata,
  but deliberately contains no machine, temperature or filament profile.
  Keep the volumes together; do not split them into separate objects or let
  the slicer drop each volume to the bed. Their shared alignment is required.
  The decorative columns and sign posts remain permanent parts of the print.
  Enable supports for the upright sculpture and upright plaque. Inspect
  the sliced preview under the torus, arrows, floating ball and sign.
  Use one of the two loaded filaments for supports; no third tool is required.
  Start with 0.12-0.16 mm layers, 3-4 perimeters and 15-20% infill, adjusting
  for your filament. Review fine lettering and the 0.88 mm wireframe ribs.

EASIER PLAQUE OPTION
  optional-flat-plaque/01-panel-black-white.3mf: 160 x 168 x 8.8 mm.
  optional-flat-plaque/02-stand-BLACK-tool2.stl: 168 x 56 x 8 mm.
  Print the panel flat as supplied, relief upward, with both tools. Print
  the stand black, slot upward. These orientations do not need supports.
  Glue the panel into the stand; the seat has 0.25 mm clearance per side.

STL FALLBACK
  STL cannot reliably store material assignments. The stl-parts folder
  therefore contains separate, aligned WHITE and BLACK files for each model.
  Select BOTH matching files in one import and accept loading them as one
  object with multiple parts. Assign WHITE to tool 1 and BLACK to tool 2.
  Center and move the complete object together. Never auto-arrange its parts.
  Each color can contain multiple closed shells joined by the other material.

VERIFICATION
  Each saved STL was reloaded and checked for closed shells, consistent
  winding and positive volume. The color partition was checked for overlaps,
  missing geometry and fit. 3MF face ranges, tool assignments, colors and
  volume geometry were checked after writing. See mesh-validation.json.
  Derived locally from the saved Onshape FeatureScript geometry. No G-code
  is supplied. Final support layout and filament settings require slicing;
  these files have not been physically print-tested.

REFERENCES
  Prusa XL dimensions: https://www.prusa3d.com/p/prusa-xl-business/
  PrusaSlicer 3MF metadata: https://github.com/prusa3d/PrusaSlicer/blob/version_2.9.0/src/libslic3r/Format/3mf.cpp
  Support settings: https://help.prusa3d.com/article/support-material_1698

LICENSE
  Artwork: functor.systems identity repository, CC BY-NC-SA; see LICENSE.txt.
  Attribution: https://functor.systems
'''
(OUT / "PRINTING.txt").write_text(guide)
(OUT / "LICENSE.txt").write_text((CAD.parent / "LICENSE").read_text())
(OUT / "mesh-validation.json").write_text(json.dumps(REPORT, indent=2) + "\n")
archive = CAD / "functor-black-white-prusa-xl.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            z.write(path, str(Path("functor-black-white-prusa-xl") / path.relative_to(OUT)))
print("DONE", archive, flush=True)
