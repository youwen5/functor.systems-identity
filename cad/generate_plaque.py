"""Build a colored Onshape relief plaque from the repository's canonical artwork.

Run with Python, svgpathtools, Shapely, and Pillow. The wordmark is traced from
the supplied PNG because the SVG's embedded font URLs point to localhost.
"""
from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from svgpathtools import parse_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cad"
SCALE = 0.2  # millimetres per SVG unit; plaque width = 160 mm


def coordinates(path):
    """Sample curves more finely than the final 0.035 mm simplification."""
    result = []
    for segment in path:
        count = max(1, math.ceil(segment.length() / 1.0))
        result.extend((segment.point(t / count).real, segment.point(t / count).imag)
                      for t in range(count))
    result.append((path.end.real, path.end.imag))
    return result


def to_mm(shape):
    return affinity.affine_transform(shape, [SCALE, 0, 0, -SCALE, -100, 188])


def polygons(shape):
    if shape.geom_type == "Polygon":
        return [shape]
    return [p for p in shape.geoms if p.geom_type == "Polygon"]


def clean(shape, tolerance=0.035):
    shape = shape.simplify(tolerance, preserve_topology=True)
    # Remove only sub-resolution islands/holes, avoiding sketch slivers.
    return unary_union([Polygon(p.exterior, [h.coords for h in p.interiors
                                             if Polygon(h).area > 0.008])
                        for p in polygons(shape) if p.area > 0.012])


root = ET.parse(ROOT / "assets/canonical/hi-res-text.svg").getroot()
wire = []
arrows = {}
ring = None
for element in root.iter():
    if element.tag.split("}")[-1] != "path":
        continue
    style = element.get("style", "")
    if "stroke-width:" not in style:
        continue
    path = parse_path(element.get("d"))
    points = coordinates(path)
    width = float(re.search(r"stroke-width:\s*([\d.]+)", style)[1])
    stroke = LineString(points).buffer(width / 2, cap_style="flat", join_style="round")
    if width == 3:
        wire.append(stroke)
    elif width == 12:
        # The SVG clips a 12-unit stroke to the inside of this circle.
        ring = stroke.intersection(Polygon(points))
    elif width == 14:
        rgb = tuple(map(int, re.search(r"stroke: rgb\((\d+), (\d+), (\d+)\)", style).groups()))
        # SVG auto-start-reverse marker: its x-axis points toward the tip.
        tip = np.array(points[0])
        end = np.array(points[-1])
        axis = (tip - end) / np.linalg.norm(tip - end)
        perpendicular = np.array([-axis[1], axis[0]])
        marker = [(0.5, 0.5), (3, 3), (0.5, 5.5), (0, 5), (2, 3), (0, 1)]
        factor = 14 * 8.5 / 6  # preserveAspectRatio=xMidYMid meet
        head = Polygon([tip + factor * ((x - 2) * axis + (y - 3) * perpendicular)
                        for x, y in marker])
        arrows[rgb] = stroke.union(head)

# All rendered text is below SVG y=750. Trace row runs at the PNG's native
# resolution, preserving the Redaction / Redaction 35 letterforms and the dot.
image = Image.open(ROOT / "assets/canonical/hi-res-text.png").convert("RGB")
pixels = np.array(image)
factor = 1000 / image.width
runs = []
for y in range(int(750 / factor), image.height):
    black = np.all(pixels[y] < 128, axis=1)
    transitions = np.diff(np.r_[False, black, False].astype(int))
    for start, stop in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0]):
        runs.append(box(start * factor, y * factor, stop * factor, (y + 1) * factor))
text = unary_union(runs)

# Visible layers meet at boundaries; color solids never overlap in volume.
ring = to_mm(ring)
arrows = {rgb: clean(to_mm(shape).difference(ring)) for rgb, shape in arrows.items()}
wire = clean(to_mm(unary_union(wire)).difference(unary_union(list(arrows.values()))))
layers = [
    ("Torus wireframe", (18, 20, 24), 1.2, wire),
    ("Limit point ring", (18, 20, 24), 2.8, clean(ring)),
    ("Redaction wordmark", (18, 20, 24), 1.8, clean(to_mm(text), 0.045)),
]
for rgb, shape in arrows.items():
    name = {(172, 47, 59): "Red projection", (15, 98, 254): "Blue projection",
            (66, 190, 101): "Green projection"}[rgb]
    layers.append((name, rgb, 2.4, shape))


def loops(shape):
    return [[[round(x, 4), round(y, 4)] for x, y in loop.coords]
            for p in polygons(shape) for loop in [p.exterior, *p.interiors]]


data = []
for name, rgb, depth, shape in layers:
    assert shape.is_valid, name
    assert all(p.area > 0 for p in polygons(shape)), name
    data.append({"name": name, "rgb": list(rgb), "depth": depth,
                 "loops": loops(shape)})
    print(name, "polygons", len(polygons(shape)), "loops", len(data[-1]["loops"]),
          "vertices", sum(len(loop) for loop in data[-1]["loops"]),
          "area mm2", round(shape.area, 2))

template = '''FeatureScript 3112;
import(path : "onshape/std/common.fs", version : "3112.0");

// Canonical functor.systems artwork, CC BY-NC-SA; see repository LICENSE.
// Generated by cad/generate_plaque.py. Coordinates are millimetres.
const ART = __ART__;

function finishPart(context is Context, bodies is Query, name is string, rgb is array)
{
    setProperty(context, {"entities" : bodies, "propertyType" : PropertyType.NAME, "value" : name});
    setProperty(context, {"entities" : bodies, "propertyType" : PropertyType.APPEARANCE,
            "value" : color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)});
}

function extrudeLoops(context is Context, id is Id, loops is array,
                      origin is Vector, normal is Vector, xaxis is Vector,
                      depth is ValueWithUnits, unit is ValueWithUnits) returns Query
{
    var sketch = newSketchOnPlane(context, id + "profile", {
                "sketchPlane" : plane(origin, normal, xaxis)});
    for (var i = 0; i < size(loops); i += 1)
    {
        var points = [];
        for (var point in loops[i])
            points = append(points, vector(point[0], point[1]) * unit);
        skPolyline(sketch, "loop" ~ i, {"points" : points});
    }
    skSolve(sketch);
    opExtrude(context, id + "solid", {
                "entities" : qSketchRegion(id + "profile", true),
                "direction" : normal,
                "endBound" : BoundingType.BLIND,
                "endDepth" : depth});
    opDeleteBodies(context, id + "removeSketch", {
                "entities" : qCreatedBy(id + "profile", EntityType.BODY)});
    return qCreatedBy(id + "solid", EntityType.BODY);
}

annotation { "Feature Type Name" : "FunctorDeskPlaque",
             "Feature Type Description" : "Colored relief of the canonical functor.systems logo on a freestanding desk plaque." }
export const functorDeskPlaque = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Plaque width" }
        isLength(definition.width, { (millimeter) : [100, 160, 300] } as LengthBoundSpec);
    }
    {
        var unit = definition.width / 160;
        // A softly rounded backing, front at Y=-3; artwork faces -Y.
        var panel = extrudeLoops(context, id + "backing", __PANEL__,
                vector(0, 3, 0) * unit, vector(0, -1, 0), vector(1, 0, 0), 6 * unit, unit);
        finishPart(context, panel, "Warm white backing - 160 x 168 x 6 mm", [244, 242, 234]);

        // Flat, broad plinth: 168 x 56 x 8 mm, with an exact pocket for the panel.
        var foot = extrudeLoops(context, id + "foot", __FOOT__,
                vector(0, 0, 0) * unit, vector(0, 0, 1), vector(1, 0, 0), 8 * unit, unit);
        opBoolean(context, id + "seat", {"tools" : panel, "targets" : foot,
                "operationType" : BooleanOperationType.SUBTRACTION, "keepTools" : true});
        finishPart(context, foot, "Graphite desk plinth - 168 x 56 x 8 mm", [34, 38, 44]);

        for (var i = 0; i < size(ART); i += 1)
        {
            var art = ART[i];
            var bodies = extrudeLoops(context, id + ("art" ~ i), art.loops,
                    vector(0, -3, 0) * unit, vector(0, -1, 0), vector(1, 0, 0),
                    art.depth * unit, unit);
            finishPart(context, bodies, art.name, art.rgb);
        }
        // Checks run in Onshape during regeneration, including during test_feature.
        var solids = qBodyType(qCreatedBy(id, EntityType.BODY), BodyType.SOLID);
        if (size(evaluateQuery(context, solids)) < 8)
            throw regenError("Expected backing, plinth, and all six artwork layers.");
        if (evVolume(context, {"entities" : solids}) <= 0 * millimeter^3)
            throw regenError("Plaque has no solid volume.");
        println("Plaque built: " ~ size(evaluateQuery(context, solids)) ~ " solids; width " ~ definition.width);
    }, { "width" : 160 * millimeter });
'''

# Small corner radii on the front silhouette and base plan; this leaves a flat
# underside and broad footprint. Values include the radius, preserving dimensions.
panel = box(-76, 10, 76, 170).buffer(4, quad_segs=8)
foot = box(-80, -24, 80, 24).buffer(4, quad_segs=8)
code = template.replace("__ART__", json.dumps(data, separators=(",", ":")))
code = code.replace("__PANEL__", json.dumps(loops(panel), separators=(",", ":")))
code = code.replace("__FOOT__", json.dumps(loops(foot), separators=(",", ":")))
(OUT / "functor-desk-plaque.fs").write_text(code)

# A front orthographic proof from the same polygons that feed the CAD model.
preview = Image.new("RGB", (1200, 1300), "#e2e5e8")
draw = ImageDraw.Draw(preview)
def draw_shape(shape, fill):
    for p in polygons(shape):
        points = lambda ring: [(600 + x * 6, 1180 - y * 6) for x, y in ring.coords]
        draw.polygon(points(p.exterior), fill=fill)
        for hole in p.interiors:
            draw.polygon(points(hole), fill=(244, 242, 234))
draw_shape(box(-84, 0, 84, 8), (34, 38, 44))
draw_shape(panel, (244, 242, 234))
for name, rgb, depth, shape in layers:
    draw_shape(shape, rgb)
preview.save(OUT / "plaque-front.png")
print("FeatureScript bytes", len(code), "written to", OUT / "functor-desk-plaque.fs")
