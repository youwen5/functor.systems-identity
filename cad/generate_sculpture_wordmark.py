"""Enlarge the canonical traced wordmark for the sculpture's freestanding sign."""
import json
import re
from pathlib import Path

CAD = Path(__file__).resolve().parent
plaque = (CAD / "functor-desk-plaque.fs").read_text()
art = json.loads(re.search(r"const ART = (.*);", plaque)[1])
source = next(item["loops"] for item in art if item["name"] == "Redaction wordmark")
min_z = min(point[1] for loop in source for point in loop)
max_z = max(point[1] for loop in source for point in loop)
scale = 1.30
offset = 44 - (max_z - min_z) * scale / 2
wordmark = [[[round(x * scale, 4), round((z - min_z) * scale + offset, 4)]
             for x, z in loop] for loop in source]
path = CAD / "functor-sculpture.fs"
code = path.read_text()
code = re.sub(r"const WORDMARK = .*;", "const WORDMARK = " + json.dumps(wordmark, separators=(",", ":")) + ";", code)
path.write_text(code)
print("Updated sculpture wordmark:", len(wordmark), "outlines; height", (max_z - min_z) * scale, "mm")
