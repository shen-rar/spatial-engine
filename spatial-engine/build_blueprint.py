#!/usr/bin/env python
"""python build_blueprint.py <scene_id> [--from <source_scene_id>]

Writes artifacts/blueprint.json: a line-only 3D model of the space (floor outline extruded to
ceiling height, interior walls, furniture as plain boxes), like a 3D model of a 2D blueprint.
With --from, the new scene copies the source scene's capture artifacts (camera path, collision
mesh, reference frame) and marks itself as a blueprint scene, so the viewer draws lines
instead of the photoreal splats.

Furniture: if the scene has furniture_annotations.json (pieces marked as boxes in captured
frames, see backend/reconstruction/furniture.py), --measure reads each piece's real position,
footprint and height from the source capture (needs its COLMAP model, keyframes and splats, and
a GPU) into artifacts/furniture.json, and the blueprint draws them as recognisable line shapes
(sofa, armchair, pool table, kitchen island...). Without it, furniture is drawn as plain boxes.
"""
import argparse
import json
import shutil
from pathlib import Path

from backend.config import DATA_ROOT
from backend.reconstruction.blueprint import write_blueprint

# what a blueprint scene needs from its source capture; the splats themselves are not copied
COPY = ("cameras.json", "collision_mesh.glb", "points.ply", "reference.jpg", "scene_report.json", "sim3_alignment.npy")


def measure(scene_dir, source_dir, only_new=False):
    """only_new: keep every piece already in furniture.json exactly as it is and measure only the
    annotation ids it does not have yet (adding items to a blueprint that is already right)."""
    import numpy as np
    import pycolmap

    from backend.reconstruction.blueprint import build_blueprint
    from backend.reconstruction.furniture import measure_furniture
    from backend.reconstruction.multi_model import gather

    ann = json.loads((scene_dir / "furniture_annotations.json").read_text(encoding="utf-8"))
    report = json.loads((source_dir / "artifacts" / "scene_report.json").read_text())
    rec = pycolmap.Reconstruction(report["best_model_dir"])
    sim3 = np.load(source_dir / "artifacts" / "sim3_alignment.npy")
    cams = np.array(json.loads((source_dir / "artifacts" / "cameras.json").read_text())["centers"])
    # measure in the grid the blueprint will use (it may follow the furniture measured last time)
    fur_file = scene_dir / "artifacts" / "furniture.json"
    previous = json.loads(fur_file.read_text()) if fur_file.exists() else None
    bp = build_blueprint(source_dir / "artifacts", cams, previous)
    yaw = bp["yaw_rad"]
    kept = previous if (only_new and previous) else []
    known = {p["id"] for p in kept}
    # every registered frame, including those of sibling models merged for training (multi_model)
    frames, _, _, _ = gather(report["best_model_dir"], log=lambda m: None)
    from backend.reconstruction.furniture import PLACE
    todo = [x for x in ann if x["id"] not in known]
    args = (rec, Path(report["keyframes_dir"]), sim3, source_dir / "artifacts" / "visual_splats.ply", yaw)
    # furniture first; small items are then anchored to it and to the walls (floor-level segments)
    new = measure_furniture([x for x in todo if x["type"] not in PLACE], *args, frames=frames)
    walls = [(s[0], s[2], s[3], s[5]) for s in bp["segments"] if s[6] in ("outline", "wall") and max(s[1], s[4]) < 0.05]
    new += measure_furniture([x for x in todo if x["type"] in PLACE], *args, frames=frames,
                             context={"pieces": kept + new, "walls": walls})
    # a piece marked under another id next to one already kept is that same piece
    new = [p for p in new if not any(k["type"] == p["type"] and
                                      ((k["center"][0] - p["center"][0]) ** 2 + (k["center"][1] - p["center"][1]) ** 2)
                                      < (0.15 if p["type"] in PLACE else 0.7) ** 2 for k in kept)]
    pieces = kept + new
    (scene_dir / "artifacts" / "furniture.json").write_text(json.dumps(pieces, indent=1))
    for p in new:
        print(f"  {p['id']:20s} {p['type']:16s} size {p['size']} h {p['height']}")
    if kept:
        print(f"  ({len(kept)} pieces kept as they were)")


def object_photos(scene_dir, source_dir, max_px=480, margin=0.15):
    """artifacts/objects/<id>.jpg: each annotated piece cut from the captured frame where its box
    is largest, and objects/views.json: the capture viewpoint of that frame (an index into
    cameras.json), so the viewer can show the piece and jump to where it was filmed. Needs the
    source's keyframes (not in git); without them existing photos are kept."""
    import cv2

    ann_file = scene_dir / "furniture_annotations.json"
    report_file = source_dir / "artifacts" / "scene_report.json"
    if not ann_file.exists() or not report_file.exists():
        return
    report = json.loads(report_file.read_text())
    kf = Path(report["keyframes_dir"])
    if not kf.is_dir():
        return
    out = scene_dir / "artifacts" / "objects"
    out.mkdir(exist_ok=True)
    best = {}
    for a in json.loads(ann_file.read_text(encoding="utf-8")):
        x0, y0, x1, y1 = a["bbox"]
        if (kf / a["frame"]).exists() and (x1 - x0) * (y1 - y0) > best.get(a["id"], (0,))[0]:
            best[a["id"]] = ((x1 - x0) * (y1 - y0), a)
    cams = json.loads((source_dir / "artifacts" / "cameras.json").read_text())
    names = cams.get("names")
    if names is None:  # older camera paths: the main model's frames in name order
        import pycolmap
        names = sorted(img.name for img in pycolmap.Reconstruction(report["best_model_dir"]).images.values())
    views = {}
    for pid, (_, a) in best.items():
        img = cv2.imread(str(kf / a["frame"]))
        H, W = img.shape[:2]
        x0, y0, x1, y1 = a["bbox"]
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        crop = img[int(max(0, y0 - my) * H):int(min(1, y1 + my) * H), int(max(0, x0 - mx) * W):int(min(1, x1 + mx) * W)]
        s = min(1.0, max_px / max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
        cv2.imwrite(str(out / f"{pid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if a["frame"] in names:  # cameras.json lists the registered frames in name order
            views[pid] = names.index(a["frame"])
    (out / "views.json").write_text(json.dumps(views, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_id")
    ap.add_argument("--from", dest="source", help="scene to measure (default: scene_id itself)")
    ap.add_argument("--name", help="display name in the viewer's scene switcher")
    ap.add_argument("--measure", action="store_true", help="measure annotated furniture from the source capture")
    ap.add_argument("--only-new", action="store_true", help="with --measure: keep measured pieces, add new ids only")
    a = ap.parse_args()

    scene_dir = DATA_ROOT / a.scene_id
    source_dir = DATA_ROOT / (a.source or a.scene_id)
    art = scene_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    if a.source and a.source != a.scene_id:
        for f in COPY:
            if (source_dir / "artifacts" / f).exists():
                shutil.copy(source_dir / "artifacts" / f, art / f)
    if a.measure:
        measure(scene_dir, source_dir, a.only_new)
    object_photos(scene_dir, source_dir)
    bp = write_blueprint(scene_dir, source_dir)
    meta = {"name": a.name or f"{a.source or a.scene_id} · blueprint", "style": "blueprint"}
    if a.source:
        meta["source"] = a.source
    (scene_dir / "scene.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scene": a.scene_id, **bp["counts"]}))


if __name__ == "__main__":
    main()
