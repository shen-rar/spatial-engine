#!/usr/bin/env python
"""Create a Y-up demo office (floor at Y=0) with furniture and a labelled POI database."""
import json
import numpy as np
import trimesh
from pathlib import Path

art = Path("./data/scenes/demo_office/artifacts")
art.mkdir(parents=True, exist_ok=True)
W, D, H = 10.0, 8.0, 3.0  # X, Z, Y(up)

def box(x0, z0, w, d, h, y0=0.0):
    b = trimesh.creation.box(extents=[w, h, d])
    b.apply_translation([x0 + w / 2, y0 + h / 2, z0 + d / 2])
    return b

parts = [box(0, 0, W, D, 0.02, -0.02)]                      # floor
parts += [box(0, -0.1, W, 0.1, H), box(0, D, W, 0.1, H),   # walls
          box(-0.1, 0, 0.1, D, H), box(W, 0, 0.1, D, H)]
furniture = {"desk": (2, 5.5, 2.0, 0.9, 0.75), "espresso machine": (8.2, 1.0, 0.6, 0.5, 1.0),
             "fire extinguisher": (9.5, 6.5, 0.3, 0.3, 0.6), "sofa": (5, 1.0, 2.2, 0.9, 0.8)}
pois = []
for name, (x, z, w, d, h) in furniture.items():
    parts.append(box(x, z, w, d, h))
    pois.append({"poi_id": name.replace(" ", "_"), "label": name,
                 "centroid_xyz": [x + w / 2, h / 2, z + d / 2],
                 "bbox_min_xyz": [x, 0, z], "bbox_max_xyz": [x + w, h, z + d]})
trimesh.util.concatenate(parts).export(str(art / "collision_mesh.glb"))
(art / "poi_database.json").write_text(json.dumps(pois, indent=2))
# a test fixture, not a model: the viewer's scene list leaves it out
(art.parent / "scene.json").write_text(json.dumps({"name": "Demo office", "style": "mesh", "hidden": True}, indent=2))
print("demo scene written:", art)
