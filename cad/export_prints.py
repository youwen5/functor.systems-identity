"""Generate and validate paint-ready FDM meshes from the saved CAD parameters.

Requires manifold3d, trimesh, numpy, scipy, and shapely. Units are millimetres.
These are local reconstructions of the saved FeatureScript geometry, not REST
exports from Onshape. The source hashes and dimensional checks are recorded.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile

import manifold3d as md
import numpy as np
import trimesh
from shapely.geometry import box

CAD = Path(__file__).resolve().parent
OUT = CAD / "print-ready"
OUT.mkdir(exist_ok=True)
REPORT = {"units": "millimetres", "method": "Manifold mesh reconstruction from saved FeatureScript geometry",
          "target": "General FDM, 0.4 mm nozzle, single material then paint", "files": {}}


def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def transformed(solid, origin=(0, 0, 0), x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1)):
    return solid.transform(np.column_stack([x, y, z, origin]))


def orient(solid, center, axis):
    z = np.asarray(axis, dtype=float)
    z /= np.linalg.norm(z)
    ref = np.array([1, 0, 0] if abs(z[0]) < 0.8 else [0, 1, 0])
    x = np.cross(z, ref)
    x /= np.linalg.norm(x)
    return transformed(solid, center, x, np.cross(z, x), z)


def section(loops):
    return md.CrossSection([np.asarray(loop, dtype=float) for loop in loops], md.FillRule.EvenOdd)


def front_extrude(loops, back_y, depth, outset=0):
    # Local sketch X,Y -> world X,Z; extrusion -> world -Y.
    profile = section(loops)
    if outset:
        profile = profile.offset(outset)
    return transformed(profile.extrude(depth), (0, back_y, 0),
                       (1, 0, 0), (0, 0, 1), (0, -1, 0))


def rounded_loops(hx, hy, radius, steps=12):
    pts = []
    for i in range(4):
        cx = hx - radius if i in (0, 3) else -hx + radius
        cy = hy - radius if i < 2 else -hy + radius
        for j in range(steps + 1):
            a = math.radians(i * 90 + j * 90 / steps)
            pts.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])
    return [pts]


def rounded_solid(hx, hy, radius, z, depth):
    return section(rounded_loops(hx, hy, radius)).extrude(depth).translate((0, 0, z))


def cube(low, high):
    return md.Manifold.cube(np.subtract(high, low)).translate(low)


def cylinder_between(start, end, radius, top_radius=None):
    direction = np.subtract(end, start)
    cylinder = md.Manifold.cylinder(float(np.linalg.norm(direction)), radius,
                                   radius if top_radius is None else top_radius, 96)
    return orient(cylinder, start, direction)


def torus(center, axis, major, minor):
    # <=0.02 mm chordal deviation for the tube and the orbit, including ribs.
    def segments(radius):
        return max(16, int(math.ceil(math.pi / math.acos(max(-1, 1 - 0.015 / radius)) / 4) * 4))
    profile = md.CrossSection.circle(minor, segments(minor)).translate((major, 0))
    return orient(profile.revolve(segments(major + minor)), center, axis)


def as_trimesh(solid):
    mesh = solid.to_mesh64()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def save(name, solid, expected_components=1, expected_size=None, note=""):
    print("Meshing", name, flush=True)
    # Remove sub-micron Boolean slivers before quantizing STL vertices to float32.
    solid = solid.simplify(0.0002)
    assert solid.status() == md.Error.NoError, (name, solid.status())
    mesh = as_trimesh(solid)
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
        raw = solid.to_mesh64()
        unmerged = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties)[:, :3], faces=np.asarray(raw.tri_verts), process=False)
        print("Mesh diagnostic", name, {"watertight":mesh.is_watertight,"winding":mesh.is_winding_consistent,"volume":mesh.volume,
            "merged_vertices":len(mesh.vertices),"raw_vertices":len(unmerged.vertices),"raw_watertight":unmerged.is_watertight,
            "raw_winding":unmerged.is_winding_consistent,"min_area":float(mesh.area_faces.min())}, flush=True)
        mesh.export(CAD / "mesh-debug.ply")
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
    assert len(mesh.split(only_watertight=False)) == expected_components, (name, "disconnected solids")
    mesh.metadata["units"] = "mm"
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="stl")
    # Validate the actual binary STL, not just the higher-precision in-memory mesh.
    loaded = trimesh.load_mesh(path, process=True)
    checks = {"watertight": bool(loaded.is_watertight),
              "consistent_winding": bool(loaded.is_winding_consistent),
              "positive_volume": bool(loaded.volume > 0),
              "finite_vertices": bool(np.isfinite(loaded.vertices).all()),
              "components": len(loaded.split(only_watertight=False))}
    assert checks["watertight"] and checks["consistent_winding"] and checks["positive_volume"], (name, checks)
    assert checks["components"] == expected_components, (name, checks)
    assert loaded.bounds[0, 2] >= -0.001, (name, "below print bed")
    if expected_size is not None:
        assert np.allclose(loaded.extents, expected_size, atol=0.06), (name, loaded.extents, expected_size)
    REPORT["files"][name] = {**checks, "size_mm": np.round(loaded.extents, 4).tolist(),
        "bounds_mm": np.round(loaded.bounds, 4).tolist(), "triangles": int(len(loaded.faces)),
        "volume_mm3": round(float(loaded.volume), 4), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "print_note": note}
    print(name, checks, np.round(loaded.extents, 3), "mm", flush=True)
    return solid


plaque_source = (CAD / "functor-desk-plaque.fs").read_text()
sculpture_source = (CAD / "functor-sculpture.fs").read_text()
REPORT["source_sha256"] = {name: hashlib.sha256((CAD / name).read_bytes()).hexdigest()
    for name in ["functor-desk-plaque.fs", "functor-sculpture.fs", "onshape-model.json", "onshape-sculpture.json"]}
art = json.loads(re.search(r"const ART = (.*);", plaque_source)[1])
wordmark = json.loads(re.search(r"const WORDMARK = (.*);", sculpture_source)[1])
spec = json.loads((CAD / "onshape-sculpture.json").read_text())
assert spec["torusSupports"]["baseDiameterMm"] == 16
assert spec["sign"]["postSectionMm"] == [9, 5]
assert 'contact + 6 * unit * n' in sculpture_source
assert 'vector(0, 0, 104)' in sculpture_source

print("Building plaque", flush=True)
panel_poly = box(-76, 10, 76, 170).buffer(4, quad_segs=8)
panel = front_extrude([panel_poly.exterior.coords], 3, 6)
relief = [front_extrude(layer["loops"], -2.98, layer["depth"] + 0.02, outset=0.01) for layer in art]
panel_with_art = union([panel, *relief])
foot_poly = box(-80, -24, 80, 24).buffer(4, quad_segs=8)
foot_blank = section([foot_poly.exterior.coords]).extrude(8)
complete_plaque = union([foot_blank, panel_with_art])
save("functor-plaque-complete.stl", complete_plaque, expected_size=[168, 56, 174],
     note="Assembled one-piece version. Upright orientation; supports needed for relief overhangs. The two-piece option is easier to print.")
# Back lies on the bed; the relief faces upward. All glyphs are fused to the panel.
flat_panel = transformed(panel_with_art, (0, 0, 3), (1, 0, 0), (0, 0, -1), (0, 1, 0))
flat_panel = flat_panel.translate((0, -6, 0))
save("plaque-two-piece/01-panel-flat.stl", flat_panel, expected_size=[160, 168, 8.8],
     note="Recommended plaque face. Print as oriented, artwork upward; no supports.")
# Open-top locating recess, 0.25 mm extra clearance on each side of the panel.
slot = front_extrude([panel_poly.buffer(0.25, join_style="round").exterior.coords], 3.25, 6.5)
stand = foot_blank - slot
save("plaque-two-piece/02-stand.stl", stand, expected_size=[168, 56, 8],
     note="Print flat as oriented, slot upward, no supports. Glue panel into slot after painting; mask joint surfaces.")

print("Building sculpture", flush=True)
tilt = math.radians(spec["torus"]["forwardTiltDegrees"])
u = np.array([1., 0, 0])
n = np.array([0, -math.sin(tilt), math.cos(tilt)])
v = np.cross(n, u)
c = np.array([0., 0, spec["torus"]["centerHeightMm"]])
node = np.array([0., 0, spec["node"]["centerHeightMm"]])


def point_on_torus(theta, phi):
    theta, phi = math.radians(theta), math.radians(phi)
    return c + (43 + 14 * math.cos(phi)) * (math.cos(theta) * u + math.sin(theta) * v) + 14 * math.sin(phi) * n


base = rounded_solid(109, 72, 12, 0, 17)
cap = rounded_solid(106, 69, 11, 16.98, 2.02)
torus_body = torus(c, n, 43, 14)
ribs = []
for i in range(32):
    theta = math.radians(i * 360 / 32)
    radial = math.cos(theta) * u + math.sin(theta) * v
    tangent = -math.sin(theta) * u + math.cos(theta) * v
    ribs.append(torus(c + 43 * radial, tangent, 14, 0.44))
for i in range(12):
    phi = math.radians(i * 30)
    ribs.append(torus(c + 14 * math.sin(phi) * n, n, 43 + 14 * math.cos(phi), 0.44))
print("Joining 44 physical wireframe ribs", flush=True)
grid = union(ribs)
torus_with_grid = union([torus_body, grid])
supports = []
for theta in [225, 315]:
    contact = point_on_torus(theta, 270)
    supports.append(cylinder_between([contact[0], -8, 18], contact + 6 * n, 8, 6))
arrows = []
for theta, phi in [(225, 35), (90, 140), (315, 35)]:
    surface = point_on_torus(theta, phi)
    direction = surface - node
    direction /= np.linalg.norm(direction)
    tip = surface + 2.5 * direction
    head_base = tip - 14 * direction
    arrows.append(union([cylinder_between(node + 9.6 * direction, head_base + direction, 1.95),
                         cylinder_between(head_base, tip, 5.6, 0)]))
ball = md.Manifold.sphere(11.8, 128).translate(node)
rim = torus(node, [0, -1, 0], 11.75, 0.55)
frame_loops = [[[x, y + 44] for x, y in loop] for loop in rounded_loops(102, 18, 3)]
face_loops = [[[x, y + 44] for x, y in loop] for loop in rounded_loops(100, 16, 2)]
frame = front_extrude(frame_loops, -58, 5)
face = front_extrude(face_loops, -62.98, 0.82)
letters = front_extrude(wordmark, -63.78, 2.82, outset=0.01)
posts = [cube([x - 4.5, -63, 18], [x + 4.5, -58, 29]) for x in [-82, 82]]
sign = union([frame, face, letters, *posts])
print("Fusing sculpture, supports, pedestal and sign into one printable solid", flush=True)
sculpture = union([base, cap, torus_with_grid, *supports, *arrows, ball, rim, sign])
save("functor-sculpture-complete.stl", sculpture, expected_size=[218, 144, 202],
     note="Print upright on the broad base. Supports are required under the torus, sign and other overhangs. Use a bed larger than 218 x 144 mm and >202 mm Z capacity.")
expected_volume = spec["validation"]["solidVolumeMm3"]
measured_volume = REPORT["files"]["functor-sculpture-complete.stl"]["volume_mm3"]
REPORT["sculpture_volume_comparison"] = {"onshape_reference_mm3": expected_volume,
    "stl_mm3": measured_volume, "relative_difference": abs(measured_volume - expected_volume) / expected_volume}
assert REPORT["sculpture_volume_comparison"]["relative_difference"] < 0.005, "Geometry differs from Onshape reference"

REPORT["print_adjustments"] = [
    "Raised artwork outlines have a 0.01 mm outward offset to join point-touching pixel details and prevent non-manifold STL seams.",
    "Coincident relief/backing interfaces overlap internally by 0.02 mm before Boolean fusion, with no change to external dimensions.",
    "Plaque two-piece stand recess has 0.25 mm clearance per side for fitting and glue.",
    "All main STLs are fused into one connected closed solid; colors are intentionally omitted for painting.",
    "Curved sections are tessellated to approximately 0.015 mm chordal deviation; binary STL round-trip is validated."
]
(OUT / "mesh-validation.json").write_text(json.dumps(REPORT, indent=2) + "\n")
guide = '''FUNCTOR.SYSTEMS — PRINT FILES
Units: millimetres. Import at 100% scale.
Target: general FDM, 0.4 mm nozzle, single filament, paint afterward.

RECOMMENDED PLAQUE PRINT (TWO PIECES)
  plaque-two-piece/01-panel-flat.stl: 160 x 168 x 8.8 mm.
  plaque-two-piece/02-stand.stl: 168 x 56 x 8 mm.
  Both files are already oriented for printing. Keep the artwork / slot upward.
  Supports are not needed for these two files. Glue the panel into the stand.
  The stand includes 0.25 mm fitting clearance per side. Fit before painting,
  and mask the joint faces so paint does not consume the clearance.

ONE-PIECE PLAQUE
  functor-plaque-complete.stl: 168 x 56 x 174 mm, assembled and fused.
  Print upright as supplied. Relief overhangs can require supports; the flat,
  two-piece option above gives a cleaner face and uses less support material.

SCULPTURE
  functor-sculpture-complete.stl: 218 x 144 x 202 mm.
  This is the latest version: large raised sign and thick tapered supports.
  Print upright with the pedestal flat on the bed. Enable supports, inspect
  the layer preview under the torus and sign, and support other overhangs as
  needed. The angled columns are part of the design, not removable supports.
  Leave room on the bed for supports, skirt or brim beyond the model footprint.
  If your usable bed is only 220 mm wide, reduce scale slightly to leave room;
  scaling down also reduces the 0.88 mm wireframe rib diameter and fine text.

SLICER STARTING POINTS (ADJUST TO YOUR PRINTER AND FILAMENT)
  0.12–0.16 mm layers for the logo and sculpture; 0.20 mm is fine for the stand.
  3–4 walls, 15–20% infill, and at least 4 top/bottom layers.
  Use variable-width walls / thin-feature handling if your slicer supports it.
  Check support contact and removal access around the arrow shafts and ribs.
  Some tiny bitcrushed letter details may merge at a 0.4 mm nozzle.
  No printer-specific G-code is included and no physical print was tested.

FINISHING
  Remove supports, lightly sand, and prime. Paint the backing/torus/node ivory,
  the frame/base/wireframe/wordmark charcoal, and the arrows:
    blue #0f62fe; red #ac2f3b; green #42be65.

VERIFICATION AND PROVENANCE
  Every saved STL was reloaded and checked for closed/watertight edges,
  consistent winding, positive volume, one connected component, and dimensions.
  See mesh-validation.json for the measurements, triangle counts and checksums.
  Generated locally from the saved FeatureScript outlines and analytic model
  parameters because the connected Onshape MCP does not offer file export.
  The design is retained; internal coincident faces are fused away. Raised
  artwork has a 0.01 mm outline expansion to eliminate point-touching seams.
  Geometry validation is not a material-strength or printer-specific test.

SUPPORT AND LAYER-HEIGHT REFERENCES
  https://help.prusa3d.com/article/support-material_1698
  https://help.prusa3d.com/article/layers-and-perimeters_1748

LICENSE
  Artwork from the functor.systems identity repository, CC BY-NC-SA; see LICENSE
  in the repository. Attribution: https://functor.systems
'''
(OUT / "PRINTING.txt").write_text(guide)
(OUT / "LICENSE.txt").write_text((CAD.parent / "LICENSE").read_text())
archive = CAD / "functor-print-ready-stls.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            z.write(path, str(Path("functor-print-ready") / path.relative_to(OUT)))
print("DONE", archive, flush=True)
