"""
Blueprint model: a line-only 3D model of a captured space, the way a 2D floor plan looks when
it is extruded. Colours and object detail are dropped on purpose; what is kept is structure:

    outline    the captured floor footprint as a right-angled polygon, drawn at the floor and
               extruded up to the local ceiling height, with a vertical edge at every corner
    walls      interior straight walls the points support, as extruded rectangles
    objects    furniture-sized clusters of points, as plain axis-aligned boxes

Everything is snapped to the scene's dominant wall direction (Manhattan world), so the result
reads like a drafted plan rather than a scan. Works from the gravity-aligned frame (Y up,
floor at Y = 0) that the reconstruction pipeline writes.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CELL = 0.1  # metres per plan pixel


def load_structure_points(art: Path) -> np.ndarray:
    """Gaussian centres that are reasonably opaque and small (surfaces, not haze), plus the
    sparse SfM points. Either source alone is enough. With both, gaussians are kept inside the
    SfM points' extent: full-resolution training also places sharp gaussians on what the windows
    show (garden, trees), which would otherwise swell the house outline."""
    pts = []
    sfm = None
    if (art / "points.ply").exists():
        import trimesh
        sfm = np.asarray(trimesh.load(art / "points.ply").vertices)
    ply = art / "visual_splats.ply"
    if ply.exists():
        raw = ply.read_bytes()
        end = raw.index(b"end_header\n") + len(b"end_header\n")
        names = [ln.split()[-1] for ln in raw[:end].decode().splitlines() if ln.startswith("property")]
        d = np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(names))
        c = {k: d[:, i] for i, k in enumerate(names)}
        op = 1 / (1 + np.exp(-c["opacity"]))
        sc = np.exp(np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], 1)).max(1)
        xyz = np.stack([c["x"], c["y"], c["z"]], 1)
        keep = (op > 0.15) & (sc < 0.3)
        if sfm is not None and len(sfm) >= 100:
            lo, hi = np.percentile(sfm, 1, axis=0) - 0.5, np.percentile(sfm, 99, axis=0) + 0.5
            keep &= ((xyz >= lo) & (xyz <= hi)).all(1)
        pts.append(xyz[keep])
    if sfm is not None:
        pts.append(sfm)
    return np.concatenate(pts).astype(np.float64)


def dominant_angle(xz: np.ndarray) -> float:
    """Rotation (radians) that makes walls axis-aligned: maximise histogram peakiness."""
    best, best_a = -1.0, 0.0
    for a in np.radians(np.arange(0, 90, 0.5)):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = xz @ R.T
        s = sum(float((np.histogram(q[:, k], bins=np.arange(q[:, k].min(), q[:, k].max() + 0.1, 0.1))[0] ** 2).sum())
                for k in (0, 1))
        if s > best:
            best, best_a = s, a
    return best_a


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return [round(float(c * v[0] - s * v[1]), 4), round(float(s * v[0] + c * v[1]), 4)]


def furniture_grid(furniture, a0, min_turn=np.radians(10), agree=np.radians(6), share=0.5):
    """The plan grid, and the pieces' facings expressed in it. Bare white walls and glass leave
    the wall-direction estimate (a0) nearly flat, so it can lock onto a wrong angle; long pieces
    (sofas, tables, islands) stand parallel to the walls. When most of their length (`share`)
    agrees within `agree` on a direction more than `min_turn` away from a0, the grid follows
    them. A piece's facing is stored in the grid it was measured in ("plan_yaw", default a0)."""
    from backend.reconstruction.furniture import ELONGATED

    out, offs = [], []
    for p in furniture:
        p = dict(p)
        turn = p.pop("plan_yaw", a0) - a0  # re-express in the a0 grid
        p["facing_plan"] = _rot(p.get("facing_plan", [0, -1]), -turn)
        if "l_side_plan" in p:
            p["l_side_plan"] = _rot(p["l_side_plan"], -turn)
        out.append(p)
        if p["type"] in ELONGATED:
            f = p["facing_plan"]
            offs.append((((np.arctan2(f[1], f[0]) + np.pi / 4) % (np.pi / 2)) - np.pi / 4, max(p["size"])))
    if not offs:
        return a0, out
    angs, wts = np.array([o for o, _ in offs]), np.array([w for _, w in offs])
    best, best_w = 0.0, 0.0
    for c in angs:  # the offset most of the length agrees with
        w = wts[np.abs(((angs - c + np.pi / 4) % (np.pi / 2)) - np.pi / 4) < agree].sum()
        if w > best_w:
            best, best_w = c, w
    near = np.abs(((angs - best + np.pi / 4) % (np.pi / 2)) - np.pi / 4) < agree
    off = float(np.average(angs[near], weights=wts[near]))
    if best_w < share * wts.sum() or abs(off) < min_turn:
        return a0, out
    for p in out:  # into the new grid; pieces now within 8 degrees of it sit square to it
        f = _rot(p["facing_plan"], -off)
        ang = np.arctan2(f[1], f[0])
        rest = ((ang + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        p["facing_plan"] = _rot(f, -rest) if abs(rest) < np.radians(8) else f
        if "l_side_plan" in p:
            p["l_side_plan"] = _rot(p["l_side_plan"], -off)
    return a0 - off, out


def _rectilinear(contour: np.ndarray, min_len_px: float) -> np.ndarray:
    """Turn a pixel contour into a right-angled polygon: simplify, then alternate horizontal and
    vertical edges, dropping edges shorter than min_len_px."""
    poly = cv2.approxPolyDP(contour, 4.0, True)[:, 0, :].astype(np.float64)
    pts = [poly[0]]
    for p in poly[1:]:
        q = pts[-1]
        if abs(p[0] - q[0]) >= abs(p[1] - q[1]):
            pts.append(np.array([p[0], q[1]]))  # horizontal move
        else:
            pts.append(np.array([q[0], p[1]]))  # vertical move
    pts = np.array(pts)
    # merge collinear runs and drop tiny jogs until stable
    for _ in range(10):
        out = []
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            if np.linalg.norm(b - a) < 1e-6:
                continue
            collinear = (abs(a[0] - b[0]) < 1e-6 and abs(b[0] - c[0]) < 1e-6) or \
                        (abs(a[1] - b[1]) < 1e-6 and abs(b[1] - c[1]) < 1e-6)
            if not collinear:
                out.append(b)
        out = np.array(out)
        if len(out) < 4:
            return pts
        # snap short edges away: move the shorter edge's end onto the previous corner line
        changed = False
        for i in range(len(out)):
            a, b = out[i], out[(i + 1) % len(out)]
            if 0 < np.linalg.norm(b - a) < min_len_px:
                j = (i + 1) % len(out)
                if abs(a[0] - b[0]) < 1e-6:  # vertical jog: pull b's y back to a's
                    out[j][1] = a[1]
                    out[(j + 1) % len(out)][1] = a[1]
                else:
                    out[j][0] = a[0]
                    out[(j + 1) % len(out)][0] = a[0]
                changed = True
                break
        pts = out
        if not changed:
            break
    return pts


def build_blueprint(art: Path, cameras: np.ndarray | None = None, furniture: list | None = None) -> dict:
    """furniture: measured pieces (see furniture.py), drawn as recognisable line templates;
    without it, furniture-sized point clusters are drawn as plain boxes."""
    p = load_structure_points(art)
    p = p[(p[:, 1] > -0.3) & (p[:, 1] < 9.0)]
    xz = p[:, [0, 2]]
    a = dominant_angle(xz[(p[:, 1] > 0.3) & (p[:, 1] < 3.0)])
    if furniture:
        a, furniture = furniture_grid(furniture, a)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = xz @ R.T
    lo = np.percentile(q, 0.5, axis=0) - 1.0
    hi = np.percentile(q, 99.5, axis=0) + 1.0
    W, H = (np.ceil((hi - lo) / CELL)).astype(int)
    to_px = lambda v: ((v - lo) / CELL)  # noqa: E731  plan coords -> pixel (col, row)
    from_px = lambda px: px * CELL + lo  # noqa: E731

    def raster(sel):
        ij = to_px(q[sel]).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        img = np.zeros((H, W), np.float32)
        np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1)
        return img

    y = p[:, 1]
    occ_all = raster(np.ones(len(p), bool))

    # ---- footprint: where anything was seen, plus where the camera walked; closed and filled
    foot = (cv2.GaussianBlur(occ_all, (0, 0), 1.5) > 0.6).astype(np.uint8)
    if cameras is not None and len(cameras):
        cq = cameras[:, [0, 2]] @ R.T
        for c in to_px(cq).astype(int):
            cv2.circle(foot, tuple(int(v) for v in c), int(0.8 / CELL), 1, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.6 / CELL) | 1,) * 2)
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, k)
    foot = cv2.morphologyEx(foot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.9 / CELL) | 1,) * 2))
    cnts, _ = cv2.findContours(foot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) * CELL * CELL > 4.0]

    def height_near(x_px, y_px, r_px=8):
        """Local ceiling height: high percentile of point heights within r of a plan pixel."""
        c = from_px(np.array([x_px, y_px], dtype=np.float64))
        d = np.abs(q - c).max(1)
        near = y[d < r_px * CELL]
        if len(near) < 15:
            return 3.0
        return float(np.clip(np.percentile(near, 95), 2.7, 8.0))

    def to_world(px, py, yy):
        v = from_px(np.array([px, py], dtype=np.float64)) @ R  # inverse rotation
        return [round(float(v[0]), 3), round(float(yy), 3), round(float(v[1]), 3)]

    segs = []  # [x1, y1, z1, x2, y2, z2] per kind

    def add(kind, p1, p2, obj=None):
        segs.append({"k": kind, "a": p1, "b": p2, "o": obj})

    outlines = []
    for c in cnts:
        poly = _rectilinear(c, min_len_px=1.2 / CELL)
        outlines.append(poly)
        n = len(poly)
        hts = [height_near(*poly[i]) for i in range(n)]
        edge_h = [min(hts[i], hts[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            j = (i + 1) % n
            add("outline", to_world(*poly[i], 0), to_world(*poly[j], 0))
            add("outline", to_world(*poly[i], edge_h[i]), to_world(*poly[j], edge_h[i]))
            # the corner post reaches the taller of its two walls, so wall tops join up
            add("corner", to_world(*poly[i], 0), to_world(*poly[i], max(edge_h[i - 1], edge_h[i])))

    inside = np.zeros((H, W), np.uint8)
    for poly in outlines:
        cv2.fillPoly(inside, [np.round(poly).astype(np.int32)], 1)
    border = cv2.dilate(inside, np.ones((3, 3), np.uint8)) - cv2.erode(inside, np.ones((9, 9), np.uint8))

    # ---- interior walls: tall, thin, straight point structures inside the footprint
    tall = raster((y > 0.4) & (y < 2.4))
    walls = ((cv2.GaussianBlur(tall, (0, 0), 0.8) > 1.2) & (inside > 0) & (border == 0)).astype(np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # A long line-shaped opening keeps only straight runs along one axis; each surviving
    # component that is thin across the axis becomes one wall at its median offset.
    run = int(1.6 / CELL)
    cand = []  # (score, horizontal?, offset, start, end)  in plan pixels
    for horiz, kern in ((True, np.ones((1, run), np.uint8)), (False, np.ones((run, 1), np.uint8))):
        op = cv2.morphologyEx(walls, cv2.MORPH_OPEN, kern)
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(op)
        for i in range(1, n_lab):
            x, yy, w, h, area = stats[i]
            length, thick = (w, h) if horiz else (h, w)
            if thick * CELL > 0.7 or length < run:
                continue
            rows, cols = np.nonzero(lab == i)
            off = float(np.median(rows if horiz else cols))
            s0, s1 = (x, x + w) if horiz else (yy, yy + h)
            cand.append((length * area / max(thick, 1), horiz, off, float(s0), float(s1)))
    # non-maximum suppression: a weaker wall parallel and close to a stronger one is the same wall
    merged = []
    for sc_, horiz, off, s0, s1 in sorted(cand, reverse=True):
        dup = False
        for m in merged:
            if m[0] == horiz and abs(m[1] - off) < 0.8 / CELL and min(m[3], s1) - max(m[2], s0) > -0.4 / CELL:
                m[2], m[3] = min(m[2], s0), max(m[3], s1)
                dup = True
                break
        if not dup and len(merged) < 30:
            merged.append([horiz, off, s0, s1])
    merged = [(s0, off, s1, off) if horiz else (off, s0, off, s1) for horiz, off, s0, s1 in merged]
    if furniture:
        # a "wall" through a known piece of furniture is that furniture's points, not a wall
        from backend.reconstruction.furniture import FURNITURE, plan_box, separate
        boxes_px = [(*to_px(np.array(b[:2])), *to_px(np.array(b[2:])))
                    for b in (plan_box(p, a, grow=0.3) for p in separate(furniture, a) if p["type"] in FURNITURE)]

        # pieces measured before plan_yaw existed keep the original rule (any touch), so blueprints
        # already built do not change
        legacy = [("plan_yaw" not in p) for p in separate(furniture, a) if p["type"] in FURNITURE]

        def hits(s):
            # mostly inside a piece: that piece's own points. A piece standing against a real wall
            # (a console, a sofa's back) only touches a short stretch of it
            x0, z0, x1, z1 = min(s[0], s[2]), min(s[1], s[3]), max(s[0], s[2]), max(s[1], s[3])
            length = max(x1 - x0, z1 - z0, 1e-6)
            for b, old in zip(boxes_px, legacy):
                if x0 <= b[2] and x1 >= b[0] and z0 <= b[3] and z1 >= b[1]:
                    inside = max(min(x1, b[2]) - max(x0, b[0]), min(z1, b[3]) - max(z0, b[1]), 0.0)
                    if old or inside >= 0.5 * length:
                        return True
            return False
        merged = [s for s in merged if not hits(s)]
    for x1, y1, x2, y2 in merged:
        h = min(height_near(x1, y1), height_near(x2, y2), 3.2)
        a0, a1 = to_world(x1, y1, 0), to_world(x2, y2, 0)
        b0, b1 = to_world(x1, y1, h), to_world(x2, y2, h)
        add("wall", a0, a1)
        add("wall", b0, b1)
        add("wall", a0, b0)
        add("wall", a1, b1)

    # ---- objects: measured furniture as recognisable templates, else low point clusters as boxes
    if furniture:
        from backend.reconstruction.furniture import describe, draw_furniture
        lines, drawn = draw_furniture(furniture, a)
        for s in lines:
            add("object", s[0:3], s[3:6], s[7])
        objects, boxes = [{**p, **describe(p)} for p in drawn], len(drawn)
    else:
        low_sel = (y > 0.12) & (y < 2.0)
        low = raster(low_sel)
        obj = ((cv2.GaussianBlur(low, (0, 0), 1.0) > 1.0) & (inside > 0) & (border == 0)).astype(np.uint8)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        obj[walls > 0] = 0
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
        qpx = to_px(q).astype(int)
        found = []
        for i in range(1, n_lab):
            x, yy, w, h, area = stats[i]
            if area * CELL * CELL < 0.5 or w * CELL > 3.5 or h * CELL > 3.5:
                continue
            member = low_sel & (qpx[:, 0] >= x) & (qpx[:, 0] < x + w) & (qpx[:, 1] >= yy) & (qpx[:, 1] < yy + h)
            if member.sum() < 30:
                continue
            found.append((w * h, x, yy, w, h, float(np.clip(np.percentile(y[member], 90), 0.35, 2.0))))
        # largest first; a box mostly inside an earlier one is the same piece of furniture
        kept = []
        for _, x, yy, w, h, top in sorted(found, reverse=True):
            inter = [max(0, min(x + w, X + Wd) - max(x, X)) * max(0, min(yy + h, Y + Hd) - max(yy, Y))
                     for X, Y, Wd, Hd, _ in kept]
            if any(v > 0.3 * w * h for v in inter):
                continue
            kept.append((x, yy, w, h, top))
        objects = []
        for i, (x, yy, w, h, top) in enumerate(kept):
            c = to_world(x + w / 2, yy + h / 2, 0)
            objects.append({"center": [c[0], c[2]], "size": [round(w * CELL, 2), round(h * CELL, 2)],
                            "height": round(top, 2), "name": f"Object {i + 1}",
                            "dims": {"width": round(w * CELL, 2), "depth": round(h * CELL, 2), "height": round(top, 2)},
                            "measured": ["width", "depth", "height"]})
            cs = [(x, yy), (x + w, yy), (x + w, yy + h), (x, yy + h)]
            for j in range(4):
                c0, c1 = cs[j], cs[(j + 1) % 4]
                add("object", to_world(*c0, 0), to_world(*c1, 0), i)
                add("object", to_world(*c0, top), to_world(*c1, top), i)
                add("object", to_world(*c0, 0), to_world(*c0, top), i)
        boxes = len(kept)

    xs = np.array([s["a"] for s in segs] + [s["b"] for s in segs]) if segs else np.zeros((1, 3))
    return {
        "style": "blueprint",
        "yaw_rad": float(a),
        "floor_y": 0.0,
        "bounds": {"min": xs.min(0).round(3).tolist(), "max": xs.max(0).round(3).tolist()},
        "counts": {"outline_corners": int(sum(len(o) for o in outlines)), "walls": len(merged), "objects": boxes},
        "objects": objects,
        # x1 y1 z1 x2 y2 z2 kind [object index]: the index ties a furniture line to its piece
        "segments": [s["a"] + s["b"] + [s["k"]] + ([s["o"]] if s["o"] is not None else []) for s in segs],
    }


def write_blueprint(scene_dir: Path, source_dir: Path | None = None) -> dict:
    """Write <scene_dir>/artifacts/blueprint.json, measured from source_dir's artifacts
    (default: the scene itself), so a blueprint scene can reuse another scene's capture."""
    art = scene_dir / "artifacts"
    src = (source_dir or scene_dir) / "artifacts"
    cams = None
    if (src / "cameras.json").exists():
        cams = np.array(json.loads((src / "cameras.json").read_text())["centers"])
    furniture = None
    if (art / "furniture.json").exists():
        furniture = json.loads((art / "furniture.json").read_text())
    bp = build_blueprint(src, cams, furniture)
    # photo of each piece and the capture spot it was filmed from (build_blueprint.object_photos)
    views_file = art / "objects" / "views.json"
    views = json.loads(views_file.read_text()) if views_file.exists() else {}
    for o in bp["objects"]:
        if "id" in o and (art / "objects" / f"{o['id']}.jpg").exists():
            o["photo"] = f"objects/{o['id']}.jpg"
        if o.get("id") in views:
            o["view"] = views[o["id"]]
    art.mkdir(parents=True, exist_ok=True)
    (art / "blueprint.json").write_text(json.dumps(bp, separators=(",", ":")))
    return bp
