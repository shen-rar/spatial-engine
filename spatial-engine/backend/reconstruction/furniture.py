"""
Recognisable furniture for the blueprint model.

Each piece is marked once or more as a box in a captured frame (furniture_annotations.json).
measure_furniture() back-projects those pixels with depth rendered from the trained splats,
so position, footprint and height come from the capture, not from a guess. draw_furniture()
then draws each piece as a line template of its type (a sofa has a seat, back and arms; a pool
table has rails and pockets; a staircase has steps), snapped to the plan's wall direction.

Annotation format (bbox is x0, y0, x1, y1 as fractions of the frame):
    {"id": "armchair_1", "type": "armchair", "frame": "frame_000302.jpg",
     "bbox": [0.15, 0.64, 0.48, 0.84], "facing": "away"}
facing says where the front of the piece points, seen from that frame: camera | away | left | right.
Optional "shape" is a drawing hint: "square" (a boxy armchair), "plain" (an island without cooktop
and hood). Several annotations with the same id are pooled (better coverage of one piece).
"""
import math
from pathlib import Path

import numpy as np

# type -> (height m, (min, max) width m, depth tolerance around the piece m, typical depth m)
# Width is measured from the marked box; front-to-back depth is the type's typical depth,
# because a single view cannot see it reliably.
SPECS = {
    "sofa": (0.80, (1.6, 3.6), 1.2, 0.95),
    "sectional": (0.80, (2.2, 4.5), 1.8, 2.6),
    "armchair": (0.78, (0.85, 1.05), 0.8, 0.95),
    "coffee_table": (0.38, (0.8, 1.4), 0.9, 1.1),
    "side_table": (0.50, (0.45, 0.7), 0.6, 0.5),
    "dining_set": (0.76, (1.8, 4.0), 2.0, 2.2),
    "round_dining": (0.75, (1.6, 2.4), 1.2, 1.9),
    "pool_table": (0.80, (2.3, 2.7), 1.6, 1.4),
    "kitchen_island": (0.92, (1.8, 3.6), 1.6, 1.1),
    "counter": (0.92, (1.2, 4.5), 1.5, 0.65),
    "bar_counter": (1.05, (1.2, 3.6), 1.4, 0.8),
    "stool": (0.95, (0.42, 0.5), 0.6, 0.45),
    "tv": (0.0, (1.0, 2.0), 0.8, 0.1),
    "staircase": (3.2, (2.5, 4.5), 2.2, 1.1),
    "outdoor_sofa": (0.75, (1.6, 3.0), 1.2, 0.85),
    "outdoor_armchair": (0.75, (0.7, 0.9), 0.8, 0.8),
}
FURNITURE = set(SPECS)  # the pieces that stand on the floor and get nudged apart

# Smaller items: same fields, height being the item's own (top minus its base), and where it
# sits: floor | surface (on a table, counter or shelf) | wall | ceiling (hangs) | flat (a rug).
ITEMS = {
    "plant": (1.0, (0.25, 1.6), 0.6, 0.5, "floor"),
    "floor_lamp": (1.6, (0.3, 0.7), 0.5, 0.4, "floor"),
    "table_lamp": (0.55, (0.2, 0.6), 0.4, 0.3, "surface"),
    "pendant_light": (0.5, (0.2, 1.0), 0.8, 0.4, "ceiling"),
    "chandelier": (0.9, (0.4, 1.8), 1.0, 0.9, "ceiling"),
    "vase": (0.35, (0.08, 0.6), 0.4, 0.2, "surface"),
    "decor": (0.2, (0.08, 0.9), 0.4, 0.25, "surface"),
    "sculpture": (0.5, (0.1, 1.2), 0.5, 0.3, "surface"),
    "cushion": (0.4, (0.3, 0.7), 0.3, 0.15, "surface"),
    "rug": (0.02, (1.0, 4.5), 1.5, 2.2, "flat"),
    "wall_art": (0.8, (0.3, 2.4), 0.5, 0.04, "wall"),
    "mirror": (0.9, (0.3, 1.6), 0.5, 0.04, "wall"),
    "curtain": (2.6, (0.3, 2.2), 0.6, 0.12, "floor"),
    "sideboard": (0.8, (0.8, 3.0), 0.8, 0.45, "floor"),
    "fireplace": (0.45, (0.8, 2.5), 0.6, 0.3, "wall"),
    "ottoman": (0.45, (0.4, 1.0), 0.5, 0.6, "floor"),
    "bench": (0.45, (0.8, 2.2), 0.6, 0.45, "floor"),
    "refrigerator": (2.0, (0.7, 1.8), 0.8, 0.7, "floor"),
    "shelf": (1.8, (0.6, 3.0), 0.8, 0.35, "floor"),
    "piano": (1.0, (1.2, 1.9), 0.9, 1.6, "floor"),
}
SPECS.update({t: v[:4] for t, v in ITEMS.items()})
PLACE = {t: v[4] for t, v in ITEMS.items()}
# heights (m) the points of an item may have when it is measured
BANDS = {"floor": (0.06, 2.6), "surface": (0.3, 2.2), "wall": (0.3, 3.6), "ceiling": (1.4, 6.0), "flat": (-0.15, 0.12)}
# round or square in plan: their depth is their width
ROUND = {"coffee_table", "armchair", "stool", "side_table", "plant", "floor_lamp", "table_lamp", "pendant_light",
         "chandelier", "vase", "decor", "sculpture", "ottoman"}


# ------------------------------------------------------------------------------- measuring
def _piece_layer(d, uu, vv, box, band=0.5, min_share=0.05, nearest=False):
    """Depth (m) of the layer an annotated box was drawn around. The box is drawn tight around
    the piece, so the piece's layer spans the box both across and top to bottom; an occluder in
    front only covers part of it (chairs before a sofa cover its lower middle) and background
    only shows through gaps (window above a sofa back). Each peak of the depth histogram is
    scored by the share of the box's columns times the share of its rows it reaches. Returns the
    layer's depth and its score (how clearly the piece shows in this view, up to ~1).
    nearest: a small item (vase, pendant, frame) is in front of whatever shows round it, so take
    the nearest layer holding a fair share of the box instead of the one that fills it most."""
    u0, v0, u1, v1 = box
    hist, edges = np.histogram(d, bins=np.arange(0, d.max() + 0.2, 0.2))
    near = np.convolve(hist, np.ones(3), "same")
    if nearest:
        for i in range(len(near)):
            if near[i] >= 0.15 * len(d) and near[i] >= near[min(i + 1, len(near) - 1)]:
                c = edges[i] + 0.1
                sel = np.abs(d - c) < band
                rows = len(np.unique(vv[sel])) / max(1, v1 - v0)
                cols = len(np.unique(uu[sel])) / max(1, u1 - u0)
                return c, rows * cols
    best, best_score = edges[int(np.argmax(near))] + 0.1, 0.0
    for i in range(len(near)):
        if near[i] < min_share * len(d) or near[i] < near[max(i - 1, 0)] or near[i] < near[min(i + 1, len(near) - 1)]:
            continue
        c = edges[i] + 0.1
        sel = np.abs(d - c) < band
        rows = len(np.unique(vv[sel])) / max(1, v1 - v0)
        cols = len(np.unique(uu[sel])) / max(1, u1 - u0)
        score = rows * cols + 0.05 * sel.mean()  # share breaks near ties
        if score > best_score:
            best, best_score = c, score
    return best, best_score


# long pieces whose orientation is read from their own points; square ones, flat screens (a
# thin plane of few points) and stairs (their points climb along the run; they follow a wall)
# keep the camera's
ELONGATED = {"sofa", "sectional", "dining_set", "pool_table", "kitchen_island", "counter", "bar_counter",
             "outdoor_sofa"}


def _long_axis(xz):
    """Unit direction (plan) along which a piece's points spread most, and how much more they
    spread along it than across it (1 = round)."""
    c = xz - xz.mean(0)
    w, v = np.linalg.eigh(c.T @ c / max(1, len(c)))
    return v[:, 1], float(np.sqrt(w[1] / max(w[0], 1e-9)))


def measure_furniture(annotations, rec, image_dir: Path, sim3, splat_ply: Path, yaw: float,
                      width=480, device="cuda", frames=None, context=None):
    """Return one measured piece per annotation id:
    {id, type, center [x, z], size [along plan-x, along plan-z], height, facing [fx, fz], l_corner,
    base (items off the floor)}. frames: gathered frames (multi_model.gather), so pieces can be
    marked in frames of merged sibling models too; default: the model's own frames.
    context: {"pieces": furniture already measured (world centres), "walls": floor-level wall
    segments (x1, z1, x2, z2)}, which small items are anchored to (see anchor_depth)."""
    import torch

    from backend.reconstruction.splat_trainer import load_colmap_views, load_ply, render

    if frames is not None:
        from backend.reconstruction.multi_model import load_views
        views, K = load_views(frames, image_dir, sim3, width)
        by_name = {v["name"]: v for v in views}
    else:
        views, K = load_colmap_views(rec, image_dir, sim3, width)
        names = [img.name for img in sorted(rec.images.values(), key=lambda i: i.name)]
        by_name = dict(zip(names, views))
    sp = load_ply(splat_ply, device)
    fx, fy, cx, cy = K
    H, W = views[0]["img"].shape[:2]
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    depth_cache = {}

    def depth_of(name):
        if name not in depth_cache:
            w2c = by_name[name]["w2c"].to(device)
            with torch.no_grad():
                z = (sp.means @ w2c[:3, :3].T + w2c[:3, 3])[:, 2:3].expand(-1, 3).contiguous()
                dz = render(sp, w2c, K, H, W, values=z)[..., 0]
                al = render(sp, w2c, K, H, W, values=torch.ones_like(z))[..., 0]
            depth_cache[name] = ((dz / al.clamp(min=1e-3)).cpu().numpy(), al.cpu().numpy())
        return depth_cache[name]

    ctx_pieces = (context or {}).get("pieces", [])
    ctx_walls = (context or {}).get("walls", [])
    supports = [q for q in ctx_pieces if q["type"] in SURFACES]
    hangs_over = [q for q in ctx_pieces if q["type"] in ("dining_set", "round_dining", "coffee_table", "kitchen_island",
                                                        "piano", "pool_table", "bar_counter", "counter")]

    def anchor_depth(typ, cam, Rc, box, layer=None):
        """Camera depth of a small item from geometry that is already known, else the depth map's
        layer, or None. Thin or see-through items show mostly what is behind them in the depth
        map, so where it can, the item is tied to: the sofa or table its centre line passes over
        (on a surface), the table a light hangs over, the floor where its box's bottom edge meets
        it (only when that line points clearly down: a shallow line moves metres for centimetres),
        or the wall it hangs on. An item far beyond the furniture its line passes over stands
        just behind that furniture (a lamp or mirror behind a dining table)."""
        bx0, by0, bx1, by1 = box
        uc = (bx0 + bx1) / 2 * W
        ray = lambda vy: Rc @ np.array([(uc - cx) / fx, (vy - cy) / fy, 1.0])  # noqa: E731  depth 1
        place = PLACE[typ]
        r = ray((by0 + by1) / 2 * H)
        rx, rz = r[0], r[2]

        def over(pieces, beyond=False):
            """Depth at which the centre line passes over one of `pieces`: the crossing nearest the
            depth map's layer when there is one (a cushion on the far sofa, not on the chair its
            line crosses first), else the nearest; beyond=True: just past the far side of the
            farthest piece it crosses."""
            hits = []
            for q in pieces:
                c = np.array(q["center"])
                s_ = ((c[0] - cam[0]) * rx + (c[1] - cam[2]) * rz) / (rx * rx + rz * rz)
                if s_ > 0.3 and np.hypot(cam[0] + s_ * rx - c[0], cam[2] + s_ * rz - c[1]) <= 0.5 * max(q["size"]):
                    hits.append(s_ + (0.5 * max(q["size"]) + 0.3 if beyond else 0.0))
            if not hits:
                return None
            if beyond:
                return max(hits)
            return min(hits, key=lambda h: abs(h - layer)) if layer is not None else min(hits)

        behind = over([q for q in ctx_pieces if q["type"] in FURNITURE], beyond=True)
        if place == "surface" or typ == "plant":
            on = [q for q in supports
                  if (q["type"] in ("sofa", "sectional", "armchair", "outdoor_sofa")) == (typ == "cushion")]
            z = over(on)
            if z is not None:
                return z
        if place == "ceiling":
            z = over(hangs_over)
            return z if z is not None else layer
        if place in ("floor", "flat") or typ == "plant":
            rb = ray(by1 * H if place == "floor" else (by0 + by1) / 2 * H)
            down = -rb[1] / np.hypot(rb[0], rb[2])
            if down > math.tan(math.radians(12)) and cam[1] > 0:
                z = (0.0 - cam[1]) / rb[1]
                if 0.3 < z < 25:
                    return z
        if place == "wall":
            wall = None
            for x1_, z1_, x2_, z2_ in ctx_walls:
                ex, ez = x2_ - x1_, z2_ - z1_
                den = rx * ez - rz * ex
                if abs(den) < 1e-9:
                    continue
                s_ = ((x1_ - cam[0]) * ez - (z1_ - cam[2]) * ex) / den
                t_ = ((x1_ - cam[0]) * rz - (z1_ - cam[2]) * rx) / den
                if 0 <= t_ <= 1 and s_ > 0.5 and (wall is None or s_ < wall):
                    wall = s_
            # a frame lies flat on its wall, a big opaque surface the depth map does see: the
            # blueprint wall only refines it (it may be missing, so a line runs on to a far one)
            if wall is not None and (layer is None or abs(wall - layer) < 1.0):
                return wall
            if layer is not None:
                return layer
        cands = [z for z in (layer, behind) if z is not None]
        return min(cands) if cands else None

    pooled = {}
    for a in annotations:
        if a["frame"] not in by_name:
            continue
        typ = a["type"]
        tol = SPECS.get(typ, (0, 0, 1.2, 0))[2]
        if typ in ("vase", "decor", "cushion", "table_lamp", "sculpture"):
            tol = 0.2
        depth, alpha = depth_of(a["frame"])
        x0, y0, x1, y1 = a["bbox"]
        u0, u1, v0, v1 = int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)
        vv, uu = np.mgrid[v0:v1, u0:u1]
        d, al = depth[v0:v1, u0:u1], alpha[v0:v1, u0:u1]
        ok = al > 0.6
        uu, vv, d = uu[ok], vv[ok], d[ok]
        w2c = by_name[a["frame"]]["w2c"].numpy().astype(np.float64)
        Rc = w2c[:3, :3].T
        cam = -Rc @ w2c[:3, 3]
        pc = np.stack([(uu + 0.5 - cx) / fx * d, (vv + 0.5 - cy) / fy * d, d], 1)
        pw = pc @ Rc.T + cam
        keep = (pw[:, 1] > 0.06) & (pw[:, 1] < (4.5 if typ == "staircase" else 2.2))
        if typ == "tv":
            keep = (pw[:, 1] > 0.4) & (pw[:, 1] < 3.0)
        if typ in PLACE:
            lo_h, hi_h = BANDS[PLACE[typ]]
            keep = (pw[:, 1] > lo_h) & (pw[:, 1] < (4.0 if typ == "curtain" else hi_h))
        pw, d, uu, vv = pw[keep], d[keep], uu[keep], vv[keep]
        z_anchor = anchor_depth(typ, cam, Rc, a["bbox"]) if typ in PLACE else None  # no depth map yet
        if len(pw) < 30 and z_anchor is None:
            continue
        if len(pw) < 30:  # a small item anchored without depth points: nothing to read its shape from
            pw = (cam + z_anchor * (Rc @ np.array([((x0 + x1) / 2 * W - cx) / fx, ((y0 + y1) / 2 * H - cy) / fy, 1.0])))[None]
            d, uu, vv = np.array([z_anchor]), np.array([u0]), np.array([v0])
        # the piece is the depth layer that fills its box best; things in front of it (a coffee
        # table before a sofa, stools before an island) and background behind it go
        mode, clarity = _piece_layer(d, uu, vv, (u0, v0, u1, v1), nearest=typ in PLACE)
        # orientation from the piece's whole depth range: a sofa receding from the camera spans
        # metres of depth, and its front slice alone looks square. Of the tolerance and twice it,
        # take the range whose points show the clearer long axis
        shapes = [_long_axis(xz) for xz in (pw[np.abs(d - mode) < k * tol][:, [0, 2]] @ R.T for k in (1, 2))
                  if len(xz) >= 30]
        sel = (d > mode - 0.3) & (d < mode + tol)
        pw, d = pw[sel], d[sel]
        fwd = Rc[:, 2]
        f_plan = R @ np.array([fwd[0], fwd[2]])
        right_plan = R @ np.array([Rc[0, 0], Rc[2, 0]])
        facing = {"camera": -f_plan, "away": f_plan, "left": -right_plan, "right": right_plan}[a.get("facing", "camera")]
        e = pooled.setdefault(a["id"], {"type": typ, "centers": [], "widths": [], "tops": [], "bottoms": [],
                                        "clarity": [], "axes": [], "facing": facing, "across": right_plan,
                                        "l_side": {"left": -right_plan, "right": right_plan}.get(a.get("l_side")),
                                        "shape": a.get("shape"), "obs": [], "cut": [], "dist": []})
        # robust centre: median of the piece's own depth layer; width: the marked box at that depth
        xz_plan = pw[:, [0, 2]] @ R.T
        e["centers"].append(np.median(xz_plan, axis=0))
        e["clarity"].append(clarity)
        e["obs"].append((w2c, a["bbox"]))
        e["axes"].append(max(shapes, key=lambda s: s[1]) if shapes else _long_axis(xz_plan))
        e["widths"].append((x1 - x0) * W * float(np.median(d)) / fx)
        if "width_m" in a:  # a piece seen end-on (a long table receding) shows no width: estimate it
            e["width_override"] = float(a["width_m"])
        e["tops"].append(float(np.percentile(pw[:, 1], 95)))
        e["bottoms"].append(float(np.percentile(pw[:, 1], 5)))
        e["cut"].append(x0 <= 0.01 or x1 >= 0.99)  # the box runs off the frame: its shape is partial
        if typ in PLACE:
            # distance: anchored to the floor, the surface below, the wall or the table under a
            # hanging light when possible (thin or see-through items show mostly what is behind
            # them in the depth map); then centre, width, top and bottom from the tight box at it
            z = anchor_depth(typ, cam, Rc, a["bbox"], layer=float(np.median(d)))
            uc = (x0 + x1) / 2 * W
            at = lambda vy: Rc @ np.array([(uc - cx) / fx * z, (vy - cy) / fy * z, z]) + cam  # noqa: E731
            mid = at((y0 + y1) / 2 * H)
            e["centers"][-1] = np.array([mid[0], mid[2]]) @ R.T
            e["widths"][-1] = (x1 - x0) * W * z / fx
            e["dist"].append(z)
            ys = [float(at(vy)[1]) for vy in (y0 * H, y1 * H)]
            e["tops"][-1], e["bottoms"][-1] = max(ys), max(min(ys), 0.0)
            if PLACE[typ] in ("floor", "flat"):
                e["bottoms"][-1] = 0.0

    def snap(v):
        ax = int(np.argmax(np.abs(v)))
        out = [0.0, 0.0]
        out[ax] = float(np.sign(v[ax]) or 1.0)
        return out

    def orient(e):
        """Facing of a long piece from the spread of its own points: the front is square to its
        long axis, on the side the annotation says. Snapped to the walls when within 15 degrees.
        Square pieces, or points too round to tell, keep the camera-based facing snapped."""
        k = int(np.argmax(e["clarity"]))  # the clearest view
        axis, elong = e["axes"][k]
        if e["type"] not in ELONGATED or elong < 1.6 or e["cut"][k]:
            return snap(e["facing"])
        f = np.array([-axis[1], axis[0]])
        f = f if np.dot(f, e["facing"]) >= 0 else -f
        ang = math.degrees(math.atan2(f[1], f[0])) % 90
        if min(ang, 90 - ang) < 15:
            return snap(f)
        return [round(float(f[0]), 4), round(float(f[1]), 4)]

    out = []
    for pid, e in pooled.items():
        h_spec, (wmin, wmax), _, depth_spec = SPECS.get(e["type"], (0.8, (0.4, 3.0), 1.2, 0.8))
        if e["type"] in PLACE and len(e["dist"]) > 1:
            # an item marked in several frames: the nearest view alone (seen largest, and a capture
            # that turns between far views may not agree on where a small thing is)
            k = int(np.argmin(e["dist"]))
            for key in ("centers", "widths", "tops", "bottoms", "clarity", "axes", "cut", "obs"):
                e[key] = e[key][k:k + 1]
        # a piece seen in several frames: trust the views where it shows most clearly (an
        # almost hidden view would pull it onto whatever stood in front)
        best = max(e["clarity"])
        ctr = np.mean([c for c, q in zip(e["centers"], e["clarity"]) if q >= best - 0.1], axis=0)
        w_seen = e["widths"][int(np.argmax(e["clarity"]))] if e["type"] in PLACE else max(e["widths"])
        width = float(np.clip(e.get("width_override", w_seen), wmin, wmax))
        facing = orient(e)
        # width runs across the facing direction; size is given along plan x and plan z
        size = [depth_spec, width] if abs(facing[0]) > abs(facing[1]) else [width, depth_spec]
        if e["type"] in ROUND or e["type"] == "round_dining":
            size = [width, width] if e["type"] != "round_dining" else [depth_spec, depth_spec]
        height = max(e["tops"]) if e["type"] != "tv" else min(e["bottoms"])
        piece = {"id": pid, "type": e["type"], "center": (ctr @ R).round(3).tolist(),
                 "size": [round(v, 2) for v in size], "height": round(height, 2), "facing_plan": facing,
                 "plan_yaw": round(float(yaw), 6)}  # the grid facing_plan is expressed in
        place = PLACE.get(e["type"])
        bottom = float(np.median(e["bottoms"]))
        if place == "flat":
            piece["height"] = 0.02
        elif place in ("surface", "wall", "ceiling") or (place and bottom > 0.2):
            piece["base"] = round(bottom, 2)  # raised (a plant on a table): its bottom; "height" is its top
        if e["l_side"] is not None:
            piece["l_side_plan"] = snap(e["l_side"])
        if e["shape"]:  # drawing hint from the annotation, e.g. a square (tufted) armchair
            piece["shape"] = e["shape"]
        facings = [facing, snap(e["facing"]), list(np.asarray(e["facing"]) / np.linalg.norm(e["facing"]))]
        piece = _fit_to_boxes(piece, facings, width, depth_spec, h_spec, e["obs"], yaw, K, W, H)
        # the same piece marked under two ids (seen from both sides) is drawn once
        same = 0.15 if piece["type"] in PLACE else 0.7  # two vases on one table are two vases
        if any(o["type"] == piece["type"] and np.hypot(*(np.subtract(o["center"], piece["center"]))) < same for o in out):
            continue
        out.append(piece)
    return out


def _projected_box(piece, yaw, w2c, K, W, H, h_spec):
    """Image box (fractions x0, y0, x1, y1) of a piece's 3D box seen by camera w2c, or None."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    xz = _footprint(piece, yaw) @ R  # plan -> world
    if piece["type"] == "tv":
        y0 = piece.get("height", 1.0)
        ys = (y0, y0 + 0.5625 * max(piece["size"]))
    elif piece["type"] in PLACE:
        ys = (piece.get("base", 0.0), max(piece.get("height", 0.3), piece.get("base", 0.0) + 0.02))
    else:
        ys = (0.0, min(max(piece.get("height", 0.8), 0.3), 1.4 * h_spec))
    X = np.array([[x, y, z] for x, z in xz for y in ys])
    c = X @ w2c[:3, :3].T + w2c[:3, 3]
    if (c[:, 2] < 0.05).any():
        return None
    fx, fy, cx, cy = K
    u, v = (fx * c[:, 0] / c[:, 2] + cx) / W, (fy * c[:, 1] / c[:, 2] + cy) / H
    return u.min(), v.min(), u.max(), v.max()


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _fit_to_boxes(piece, facings, width, depth_spec, h_spec, obs, yaw, K, W, H, reach=0.5, step=0.1):
    """Fine-tune a measured piece so its 3D box, seen from each frame it was marked in, covers
    the marked box: the photoreal model reproduces those frames at their capture spots, so this
    is what makes the blueprint line up with it. Tries the candidate facings (from the piece's
    own shape, snapped to the walls, as the camera saw it) and shifts within `reach` metres,
    with a slight preference for the measured spot."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    base = np.array(piece["center"]) @ R.T
    square = piece["type"] in ROUND or piece["type"] == "round_dining"
    best, best_s = piece, -1.0
    seen = set()
    n = int(round(reach / step))
    for f in facings:
        f = [round(float(f[0]), 4), round(float(f[1]), 4)]
        if tuple(f) in seen:
            continue
        seen.add(tuple(f))
        size = piece["size"] if square else ([depth_spec, width] if abs(f[0]) > abs(f[1]) else [width, depth_spec])
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                cand = {**piece, "facing_plan": f, "size": [round(v, 2) for v in size],
                        "center": ((base + np.array([i, j]) * step) @ R).round(3).tolist()}
                ious = [_iou(b, box) if (b := _projected_box(cand, yaw, w2c, K, W, H, h_spec)) else 0.0
                        for w2c, box in obs]
                s = float(np.mean(ious)) - 0.02 * math.hypot(i, j) * step
                if s > best_s:
                    best, best_s = cand, s
    return best


# ------------------------------------------------------------------------------- drawing
class _Pen:
    """Collects line segments in plan-local coordinates of one piece and maps them to world.
    Local frame: u along the piece's width, v from front (-) to back (+), y up."""

    def __init__(self, center_plan, u_axis, v_axis, R, base=0.0):
        self.c, self.u, self.v, self.R = np.asarray(center_plan), np.asarray(u_axis), np.asarray(v_axis), R
        self.base = base  # heights are drawn from here (an item standing on a table, a frame on a wall)
        self.segs = []

    def _w(self, u, v, y):
        p = (self.c + u * self.u + v * self.v) @ self.R
        return [round(float(p[0]), 3), round(float(y + self.base), 3), round(float(p[1]), 3)]

    def line(self, a, b):
        self.segs.append(self._w(*a) + self._w(*b))

    def rect(self, u0, v0, u1, v1, y):
        for a, b in (((u0, v0), (u1, v0)), ((u1, v0), (u1, v1)), ((u1, v1), (u0, v1)), ((u0, v1), (u0, v0))):
            self.line((*a, y), (*b, y))

    def box(self, u0, v0, u1, v1, y0, y1):
        self.rect(u0, v0, u1, v1, y0)
        self.rect(u0, v0, u1, v1, y1)
        for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            self.line((u, v, y0), (u, v, y1))

    def circle(self, u, v, r, y, n=20, a0=0.0, a1=2 * math.pi, rv=None):
        rv = r if rv is None else rv
        pts = [(u + r * math.cos(a0 + (a1 - a0) * i / n), v + rv * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        for p, q in zip(pts, pts[1:]):
            self.line((*p, y), (*q, y))
        return pts

    def cylinder(self, u, v, r, y0, y1, n=20, posts=4):
        self.circle(u, v, r, y0, n)
        self.circle(u, v, r, y1, n)
        for i in range(posts):
            a = 2 * math.pi * i / posts + math.pi / 4
            self.line((u + r * math.cos(a), v + r * math.sin(a), y0), (u + r * math.cos(a), v + r * math.sin(a), y1))


def _sofa(pen, W, D, H, arms=True, cushions=True, seat=0.42):
    hw, hd = W / 2, D / 2
    back = 0.2
    arm = 0.18 if arms else 0.0
    pen.box(-hw, -hd, hw, hd, 0.0, seat)                       # base + seat
    pen.box(-hw, hd - back, hw, hd, seat, H)                   # backrest
    if arms:
        for s in (-1, 1):
            pen.box(s * hw - (arm if s > 0 else 0), -hd, s * hw + (0 if s > 0 else arm), hd - back, seat, seat + 0.2)
    if cushions:
        inner = W - 2 * arm
        n = max(1, round(inner / 0.75))
        for i in range(1, n):
            u = -hw + arm + inner * i / n
            pen.line((u, -hd, seat), (u, hd - back, seat))
            pen.line((u, hd - back, seat), (u, hd - back, H - 0.08))


def _armchair_round(pen, W, D, H, seat=0.42):
    r = min(W, D) / 2
    pen.circle(0, 0, r, 0.05, 24)
    pen.circle(0, 0, r, seat, 24)
    pen.circle(0, 0, r * 0.62, seat, 18)                       # seat cushion
    a0, a1 = math.radians(-20), math.radians(200)               # back wraps round the rear
    back = pen.circle(0, 0, r, H, 18, a0, a1)
    for p in (back[0], back[len(back) // 2], back[-1]):
        pen.line((p[0], p[1], seat), (p[0], p[1], H))


def _armchair_square(pen, W, D, H, seat=0.40):
    hw, hd = W / 2, D / 2
    for u in (-hw, hw):                                         # frame legs and arms
        pen.line((u, -hd, 0), (u, -hd, 0.62))
        pen.line((u, hd, 0), (u, hd, H))
        pen.line((u, -hd, 0.62), (u, hd, 0.62))
    pen.rect(-hw + 0.06, -hd + 0.04, hw - 0.06, hd - 0.1, seat)  # cushion
    pen.rect(-hw + 0.06, -hd + 0.04, hw - 0.06, hd - 0.1, seat + 0.12)
    pen.line((-hw, hd, H), (hw, hd, H))
    pen.line((-hw + 0.06, hd - 0.1, seat + 0.12), (-hw + 0.06, hd - 0.02, H - 0.05))
    pen.line((hw - 0.06, hd - 0.1, seat + 0.12), (hw - 0.06, hd - 0.02, H - 0.05))
    pen.line((-hw + 0.06, hd - 0.02, H - 0.05), (hw - 0.06, hd - 0.02, H - 0.05))


def _chair(pen, u, v, facing_v, s=0.46, seat=0.46, back=0.9):
    """Dining / bar chair at (u, v) whose front faces -facing_v along v."""
    hs = s / 2
    pen.rect(u - hs, v - hs, u + hs, v + hs, seat)
    for du in (-hs, hs):
        for dv in (-hs, hs):
            pen.line((u + du, v + dv, 0), (u + du, v + dv, seat))
    bv = v + hs * facing_v
    pen.line((u - hs, bv, seat), (u - hs, bv, back))
    pen.line((u + hs, bv, seat), (u + hs, bv, back))
    pen.line((u - hs, bv, back), (u + hs, bv, back))
    pen.line((u - hs, bv, back - 0.18), (u + hs, bv, back - 0.18))


def _table_rect(pen, W, D, H, leg=0.06):
    hw, hd = W / 2, D / 2
    pen.box(-hw, -hd, hw, hd, H - 0.05, H)
    for su in (-1, 1):
        for sv in (-1, 1):
            u, v = su * (hw - 0.12), sv * (hd - 0.12)
            pen.line((u, v, 0), (u, v, H - 0.05))


def draw_piece(piece, yaw):
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    ctr = np.array(piece["center"]) @ R.T
    sx, sz = piece["size"]
    f = np.array(piece.get("facing_plan", [0, -1]), dtype=float)
    # local v points to the back (opposite the facing), u to the piece's right
    v_axis = -f
    u_axis = np.array([-v_axis[1], v_axis[0]])
    W, D = (sx, sz) if abs(u_axis[0]) >= abs(u_axis[1]) else (sz, sx)  # facing may be off the wall grid
    pen = _Pen(ctr, u_axis, v_axis, R)
    t = piece["type"]
    H = max(piece.get("height", 0.8), 0.3)
    spec_h = SPECS.get(t, (0.8,))[0]

    if t in ("sofa", "outdoor_sofa"):
        D = min(max(D, 0.8), 1.1)
        _sofa(pen, W, D, min(max(H, 0.7), 0.95))
    elif t == "sectional":
        Hs = min(max(H, 0.7), 0.95)
        dep = 0.95
        hw, hd = W / 2, D / 2
        # which end the short arm of the L sits on (marked per annotation as l_side)
        side_u = float(np.sign(np.dot(np.array(piece.get("l_side_plan", u_axis), float), u_axis)) or 1)
        # long run along the back
        long_pen = _Pen(ctr + v_axis * (hd - dep / 2), u_axis, v_axis, R)
        _sofa(long_pen, W, dep, Hs, arms=False)
        # short run along one side, its back to the outside
        side_c = ctr + u_axis * side_u * (hw - dep / 2) - v_axis * (dep / 2)
        side_pen = _Pen(side_c, v_axis * -1, u_axis * side_u, R)
        _sofa(side_pen, max(D - dep, 0.8), dep, Hs, arms=False)
        pen.segs += long_pen.segs + side_pen.segs
        # arm at the free end of the long run
        pen.box(-side_u * hw - 0.0, hd - dep, -side_u * (hw - 0.18), hd, 0.42, 0.62)
    elif t == "armchair" and piece.get("shape") == "square":  # boxy (tufted) armchair
        _armchair_square(pen, W, D, min(max(H, 0.65), 0.85))
    elif t == "armchair":
        _armchair_round(pen, W, D, min(max(H, 0.65), 0.85))
    elif t == "outdoor_armchair":
        _armchair_square(pen, W, D, min(max(H, 0.65), 0.85))
    elif t == "coffee_table":
        r = min(W, D) / 2
        pen.cylinder(0, 0, r, 0.0, spec_h, posts=6)
        pen.circle(0, 0, r * 0.82, spec_h, 20)
    elif t == "side_table":
        _table_rect(pen, W, D, spec_h, leg=0.05)
    elif t == "dining_set":
        # table inset from the measured set; chairs along both long sides
        if W < D:
            W, D = D, W
            pen = _Pen(ctr, v_axis, -u_axis, R)
        tw, td = max(W - 0.3, 1.0), max(min(D - 1.0, 1.2), 0.9)
        _table_rect(pen, tw, td, spec_h)
        n = max(2, round(tw / 0.62))
        for i in range(n):
            u = -tw / 2 + tw * (i + 0.5) / n
            _chair(pen, u, -td / 2 - 0.3, -1)
            _chair(pen, u, td / 2 + 0.3, 1)
    elif t == "round_dining":
        r = min(max(min(W, D) / 2 - 0.45, 0.45), 0.7)
        pen.circle(0, 0, r, spec_h, 24)
        pen.circle(0, 0, r, spec_h - 0.04, 24)
        pen.line((0, 0, 0), (0, 0, spec_h - 0.04))
        pen.circle(0, 0, 0.25, 0.0, 12)
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            cu, cv = (r + 0.3) * math.cos(a), (r + 0.3) * math.sin(a)
            _chair(pen, cu, cv, 1 if cv > 0 else -1, s=0.48, seat=0.45, back=0.85)
    elif t == "pool_table":
        if W < D:
            W, D = D, W
            pen = _Pen(ctr, v_axis, -u_axis, R)
        W, D = min(W, 2.7), min(D, 1.5)
        hw, hd, h = W / 2, D / 2, spec_h
        pen.box(-hw, -hd, hw, hd, h - 0.15, h)                    # body + rail
        pen.rect(-hw + 0.12, -hd + 0.12, hw - 0.12, hd - 0.12, h - 0.03)  # playing surface
        for u in (-hw + 0.12, 0, hw - 0.12):                       # six pockets
            for v in (-hd + 0.12, hd - 0.12):
                pen.circle(u, v, 0.06, h, 8)
        for u in (-hw * 0.55, hw * 0.55):                          # two pedestal legs
            pen.box(u - 0.15, -hd + 0.25, u + 0.15, hd - 0.25, 0.0, h - 0.15)
    elif t == "kitchen_island":
        if W < D:
            W, D = D, W
            pen = _Pen(ctr, v_axis, -u_axis, R)
        hw, hd, h = W / 2, D / 2, spec_h
        pen.box(-hw + 0.05, -hd + 0.05, hw - 0.05, hd - 0.05, 0.0, h - 0.04)  # cabinet body
        pen.box(-hw, -hd, hw, hd, h - 0.04, h)                                 # worktop
        plain = piece.get("shape") == "plain"  # an island without a cooktop and hood over it
        if not plain:
            for k in range(4):                                                 # cooktop burners
                pen.circle(-0.2 + 0.4 * (k % 2), -0.18 + 0.36 * (k // 2), 0.09, h, 12)
            pen.rect(-0.38, -0.33, 0.38, 0.33, h)
        n = max(1, round((W - 0.1) / 0.6))                                     # cabinet doors
        for i in range(1, n):
            u = -hw + 0.05 + (W - 0.1) * i / n
            pen.line((u, -hd + 0.05, 0.1), (u, -hd + 0.05, h - 0.1))
        if not plain:
            pen.box(-0.4, -0.3, 0.4, 0.3, h + 0.9, h + 1.4)                     # extractor hood
            pen.line((0, 0, h + 1.4), (0, 0, 2.7))
    elif t in ("counter", "bar_counter"):
        if W < D:
            W, D = D, W
            pen = _Pen(ctr, v_axis, -u_axis, R)
        D = min(D, 0.9 if t == "bar_counter" else 0.7)
        hw, hd, h = W / 2, D / 2, (spec_h if t == "counter" else min(max(H, 0.9), 1.1))
        pen.box(-hw + 0.03, -hd + 0.05, hw - 0.03, hd, 0.0, h - 0.04)
        pen.box(-hw, -hd, hw, hd, h - 0.04, h)
        n = max(1, round(W / 0.6))
        for i in range(1, n):
            u = -hw + W * i / n
            pen.line((u, -hd + 0.05, 0.1), (u, -hd + 0.05, h - 0.1))
        pen.line((-hw + 0.03, -hd + 0.05, 0.1), (hw - 0.03, -hd + 0.05, 0.1))  # plinth
        if t == "bar_counter":
            pen.rect(-0.25, -0.15, 0.25, 0.15, h)                              # sink
            pen.line((0, 0.22, h), (0, 0.22, h + 0.35))                         # tap
            pen.line((0, 0.22, h + 0.35), (0, 0.05, h + 0.35))
        else:
            pen.box(-hw, hd - 0.35, hw, hd, 1.5, 2.3)                           # wall cabinets
    elif t == "stool":
        pen.circle(0, 0, 0.2, 0.75, 14)
        pen.line((0, 0, 0), (0, 0, 0.75))
        pen.circle(0, 0, 0.18, 0.0, 12)
        pen.circle(0, 0, 0.15, 0.3, 10)
    elif t == "tv":
        w = min(max(W, D), 2.0)
        y0 = min(max(piece.get("height", 1.0), 0.6), 1.6)
        pen.rect(-w / 2, 0, w / 2, 0.03, y0)
        pen.rect(-w / 2, 0, w / 2, 0.03, y0 + w * 0.5625)
        for u in (-w / 2, w / 2):
            pen.line((u, 0.03, y0), (u, 0.03, y0 + w * 0.5625))
        if piece.get("height", 1.0) < 1.2:  # a screen hung high is over a fireplace or bare wall
            pen.box(-w * 0.75, -0.42, w * 0.75, 0.03, 0.0, 0.45)                # media console
    elif t == "staircase":
        if W < D:
            W, D = D, W
            pen = _Pen(ctr, v_axis, -u_axis, R)
        top = min(max(H, 2.6), 3.6)
        n = max(8, round(top / 0.18))
        run = W / n
        hd = min(D, 1.3) / 2
        for i in range(n):
            u0 = -W / 2 + i * run
            y = top * (i + 1) / n
            pen.line((u0, -hd, y), (u0, hd, y))                               # nosing
            pen.line((u0, -hd, y), (u0 + run, -hd, y))
            pen.line((u0, hd, y), (u0 + run, hd, y))
            pen.line((u0 + run, -hd, y), (u0 + run, hd, y))
        pen.line((-W / 2, -hd, 0), (W / 2, -hd, top))                          # stringers
        pen.line((-W / 2, hd, 0), (W / 2, hd, top))
        pen.line((-W / 2, -hd, 0.9), (W / 2, -hd, top + 0.9))                  # handrail
        pen.line((-W / 2, -hd, 0), (-W / 2, -hd, 0.9))
        pen.line((W / 2, -hd, top), (W / 2, -hd, top + 0.9))
    elif t in PLACE:
        _draw_item(piece, t, W, D, ctr, u_axis, v_axis, R, pen)
    else:
        pen.box(-W / 2, -D / 2, W / 2, D / 2, 0.0, H)
    return pen.segs


def _draw_item(piece, t, W, D, ctr, u_axis, v_axis, R, pen):
    """Line drawings of the smaller items, from their base (floor, table top, wall or ceiling
    drop) up to their measured top."""
    base = piece.get("base", 0.0)
    h = max(piece.get("height", 0.3) - base, 0.05)
    p = _Pen(ctr, u_axis, v_axis, R, base)
    r = min(W, D) / 2
    if t == "plant":
        pot = min(0.35 * h, 0.45)
        p.cylinder(0, 0, r * 0.45, 0, pot, 14, 4)
        p.line((0, 0, pot), (0, 0, h * 0.8))
        for k, (y, rr) in enumerate(((pot + (h - pot) * 0.35, r), (pot + (h - pot) * 0.65, r * 0.85), (h, r * 0.45))):
            p.circle(0, 0, rr, y, 16)
            if k < 2:
                for a in (0.3, 2.4, 4.4):
                    p.line((0, 0, y - 0.12), (rr * math.cos(a), rr * math.sin(a), y))
    elif t in ("floor_lamp", "table_lamp"):
        sh = min(0.3 * h, 0.45)
        p.circle(0, 0, r * 0.6, 0.0, 12)
        p.line((0, 0, 0), (0, 0, h - sh))
        p.circle(0, 0, r, h - sh, 16)
        p.circle(0, 0, r * 0.65, h, 16)
        for a in (0.4, 2.0, 3.6, 5.2):
            p.line((r * math.cos(a), r * math.sin(a), h - sh), (0.65 * r * math.cos(a), 0.65 * r * math.sin(a), h))
    elif t == "pendant_light":
        p.cylinder(0, 0, r, 0.0, h, 14, 4)
        p.line((0, 0, h), (0, 0, h + 0.6))                       # cord up to the ceiling
    elif t == "chandelier":
        p.circle(0, 0, r, h * 0.35, 20)
        p.circle(0, 0, r * 0.6, h * 0.8, 16)
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            p.line((r * math.cos(a), r * math.sin(a), h * 0.35), (r * math.cos(a), r * math.sin(a), 0.0))
            p.line((r * math.cos(a), r * math.sin(a), h * 0.35), (0.6 * r * math.cos(a), 0.6 * r * math.sin(a), h * 0.8))
        p.line((0, 0, h * 0.8), (0, 0, h + 0.6))
    elif t == "vase":
        p.circle(0, 0, r * 0.6, 0.0, 12)
        p.circle(0, 0, r, h * 0.45, 14)
        p.circle(0, 0, r * 0.45, h, 12)
        prof = [(r * 0.6, 0.0), (r, h * 0.45), (r * 0.45, h)]
        for a in (0.5, 2.6, 4.7):
            for (r0, y0), (r1, y1) in zip(prof, prof[1:]):
                p.line((r0 * math.cos(a), r0 * math.sin(a), y0), (r1 * math.cos(a), r1 * math.sin(a), y1))
    elif t == "cushion":
        p.box(-W / 2, -D / 2, W / 2, D / 2, 0.0, h)
        p.rect(-W / 2 + 0.05, -D / 2, W / 2 - 0.05, -D / 2 + 0.01, h / 2)
    elif t == "rug":
        p.rect(-W / 2, -D / 2, W / 2, D / 2, 0.01)
        p.rect(-W / 2 + 0.12, -D / 2 + 0.12, W / 2 - 0.12, D / 2 - 0.12, 0.01)
    elif t in ("wall_art", "mirror"):
        for y in (0.0, h):                                        # frame on the wall plane
            p.line((-W / 2, 0.0, y), (W / 2, 0.0, y))
        for u in (-W / 2, W / 2):
            p.line((u, 0.0, 0.0), (u, 0.0, h))
        m = min(0.08, W / 6, h / 6)
        p.line((-W / 2 + m, 0.0, m), (W / 2 - m, 0.0, m))
        p.line((-W / 2 + m, 0.0, h - m), (W / 2 - m, 0.0, h - m))
        p.line((-W / 2 + m, 0.0, m), (-W / 2 + m, 0.0, h - m))
        p.line((W / 2 - m, 0.0, m), (W / 2 - m, 0.0, h - m))
        if t == "mirror":
            p.line((-W / 4, 0.0, h * 0.55), (0.0, 0.0, h * 0.8))
    elif t == "curtain":
        p.line((-W / 2 - 0.1, 0.0, h), (W / 2 + 0.1, 0.0, h))     # rod
        n = max(3, round(W / 0.15))
        for i in range(n + 1):
            u = -W / 2 + W * i / n
            p.line((u, 0.04 * (i % 2), 0.0), (u, 0.04 * (i % 2), h))
    elif t == "fireplace":
        p.box(-W / 2, -D / 2, W / 2, D / 2, 0.0, h)
        p.rect(-W / 2 + 0.08, -D / 2, W / 2 - 0.08, -D / 2 + 0.01, h / 2)
    elif t in ("sideboard", "shelf", "refrigerator"):
        p.box(-W / 2, -D / 2, W / 2, D / 2, 0.0, h)
        if t == "shelf":
            n = max(2, round(h / 0.35))
            for k in range(1, n):
                p.line((-W / 2, -D / 2, h * k / n), (W / 2, -D / 2, h * k / n))
        else:
            n = 2 if t == "refrigerator" else max(2, round(W / 0.5))
            for i in range(1, n):
                u = -W / 2 + W * i / n
                p.line((u, -D / 2, 0.08), (u, -D / 2, h - 0.04))
    elif t == "ottoman":
        p.cylinder(0, 0, r, 0.0, h, 20, 6)
    elif t == "bench":
        _table_rect(p, W, D, h, leg=0.05)
    else:  # decor, sculpture: a small block
        p.box(-W / 2, -D / 2, W / 2, D / 2, 0.0, h)
    pen.segs += p.segs


def plan_box(piece, yaw, grow=0.0):
    """Axis-aligned footprint (x0, z0, x1, z1) of a piece in plan coordinates."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    c = np.array(piece["center"]) @ R.T
    h = np.array(piece["size"]) / 2 + grow
    return (*(c - h), *(c + h))


def _footprint(piece, yaw):
    """Plan-frame corners (4, 2) of a piece's footprint, as draw_piece lays it out."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    c = np.array(piece["center"]) @ R.T
    v = -np.array(piece.get("facing_plan", [0, -1]), dtype=float)
    u = np.array([-v[1], v[0]])
    sx, sz = piece["size"]
    W, D = (sx, sz) if abs(u[0]) >= abs(u[1]) else (sz, sx)
    return np.array([c + su * u * W / 2 + sv * v * D / 2 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


# pieces that really stand partly under another: stools tuck under counters
TUCKS = {("stool", "kitchen_island"), ("stool", "counter"), ("stool", "bar_counter")}


def separate(pieces, yaw, gap=0.08):
    """Nudge smaller pieces out of larger ones they overlap (single-view positions are rough),
    along whichever axis needs the shortest move."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    pieces = [dict(p) for p in pieces]
    # only floor furniture is nudged: a vase belongs on its table, cushions on their sofa
    order = sorted((i for i in range(len(pieces)) if pieces[i]["type"] in FURNITURE),
                   key=lambda i: -np.prod(pieces[i]["size"]))
    for _ in range(3):
        for bi, big in enumerate(order):
            for small in order[bi + 1:]:
                if (pieces[small]["type"], pieces[big]["type"]) in TUCKS:
                    continue
                a, b = plan_box(pieces[big], yaw), plan_box(pieces[small], yaw)
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oz = min(a[3], b[3]) - max(a[1], b[1])
                if ox <= 0 or oz <= 0:
                    continue
                c = np.array(pieces[small]["center"]) @ R.T
                ca = np.array(pieces[big]["center"]) @ R.T
                if ox < oz:
                    c[0] += (ox + gap) * (1 if c[0] >= ca[0] else -1)
                else:
                    c[1] += (oz + gap) * (1 if c[1] >= ca[1] else -1)
                pieces[small]["center"] = (c @ R).round(3).tolist()
    return pieces


# what an item can stand on, and the height of that surface (m): table tops, a sofa's seat
SURFACES = {"coffee_table": None, "side_table": None, "sideboard": None, "kitchen_island": None, "counter": None,
            "bar_counter": None, "dining_set": None, "piano": None, "shelf": None, "bench": None,
            "sofa": 0.42, "sectional": 0.42, "armchair": 0.42, "outdoor_sofa": 0.42}
RESTING = {"vase", "decor", "sculpture", "table_lamp", "cushion", "plant"}


def settle(pieces, yaw):
    """Set items down on what they stand on: an item marked in one view has a rough distance,
    so its height is the least certain thing about it. A vase, bowl, lamp or plant whose centre
    lies on a table, sideboard, island or piano sits on its top; a cushion on a sofa's seat. The
    item keeps its own measured height (within twice or half its type's)."""
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    out = [dict(p) for p in pieces]
    supports = [p for p in out if p["type"] in SURFACES]
    for p in out:
        if p["type"] not in RESTING:
            continue
        c = np.array(p["center"]) @ R.T
        best = None
        for s in supports:
            if (s["type"] in ("sofa", "sectional", "armchair", "outdoor_sofa")) != (p["type"] == "cushion"):
                continue
            x0, z0, x1, z1 = plan_box(s, yaw, grow=0.1)
            if x0 <= c[0] <= x1 and z0 <= c[1] <= z1:
                top = SURFACES[s["type"]]
                if top is None:
                    top = min(max(s.get("height", SPECS[s["type"]][0]), 0.3), 1.4 * SPECS[s["type"]][0])
                if best is None or top > best:
                    best = top
        if best is None:
            continue
        own = p.get("height", 0.3) - p.get("base", 0.0)
        spec = SPECS[p["type"]][0]
        own = float(np.clip(own, 0.5 * spec, 2.0 * spec))
        p["base"], p["height"] = round(best, 2), round(best + own, 2)
    return out


def draw_furniture(pieces, yaw):
    """Line segments of every piece, each tagged with its index in the returned pieces (the
    viewer picks and highlights a piece by it), and the pieces where they are drawn."""
    pieces = settle(separate(pieces, yaw), yaw)
    segs = []
    for i, p in enumerate(pieces):
        segs += [s + ["object", i] for s in draw_piece(p, yaw)]
    return segs, pieces


NAMES = {"sofa": "Sofa", "sectional": "Sectional sofa", "armchair": "Armchair", "coffee_table": "Coffee table",
         "side_table": "Side table", "dining_set": "Dining table and chairs", "round_dining": "Round dining table",
         "pool_table": "Pool table", "kitchen_island": "Kitchen island", "counter": "Kitchen counter",
         "bar_counter": "Bar counter", "stool": "Bar stool", "tv": "Television", "staircase": "Staircase",
         "outdoor_sofa": "Outdoor sofa", "outdoor_armchair": "Outdoor armchair", "plant": "Plant",
         "floor_lamp": "Floor lamp", "table_lamp": "Table lamp", "pendant_light": "Pendant light",
         "chandelier": "Chandelier", "vase": "Vase", "decor": "Decor", "sculpture": "Sculpture", "cushion": "Cushion",
         "rug": "Rug", "wall_art": "Wall art", "mirror": "Mirror", "curtain": "Curtain", "sideboard": "Sideboard",
         "fireplace": "Fireplace", "ottoman": "Ottoman", "bench": "Bench", "refrigerator": "Refrigerator",
         "shelf": "Shelf", "piano": "Piano"}


def describe(piece):
    """Display name and dimensions (metres) of a measured piece. Width (across its front) and
    height come from the capture; depth front-to-back is the type's typical depth, as one view
    cannot see it, except for round or square pieces, whose depth is their width."""
    t = piece["type"]
    f = np.array(piece.get("facing_plan", [0, -1]), dtype=float)
    across_x = abs(f[1]) >= abs(f[0])  # facing mostly along plan z: the width runs along plan x
    sx, sz = piece["size"]
    width, depth = (sx, sz) if across_x else (sz, sx)
    square = t in ("coffee_table", "armchair", "stool", "side_table")
    # name from the annotation id, which tells pieces of one type apart: armchair_2 ->
    # "Armchair 2", tub_chair_1 -> "Tub chair 1", sofa_back -> "Sofa (back)", island -> "Kitchen island"
    base, _, num = piece["id"].rpartition("_")
    if not num.isdigit():
        base, num = piece["id"], ""
    label = NAMES.get(t, t.replace("_", " ").capitalize())
    head, _, rest = base.partition("_")
    if base in t:
        name = label
    elif head == t.split("_")[0] and rest:
        name = f"{label} ({rest.replace('_', ' ')})"
    else:
        name = base.replace("_", " ").capitalize()
    name += f" {num}" if num else ""
    dims = {"width": round(width, 2), "depth": round(depth, 2), "height": round(piece.get("height", 0.8), 2)}
    measured = ["width", "height"] + (["depth"] if square else [])
    h_spec = SPECS.get(t, (0.8,))[0]
    if t in PLACE:  # items: their own height, and how high they sit when raised
        if t in ROUND:
            measured.append("depth")
        base = piece.get("base", 0.0)
        dims["height"] = round(max(piece.get("height", 0.3) - base, 0.02), 2)
        if PLACE[t] in ("surface", "wall", "ceiling"):
            dims["above_floor"] = round(base, 2)
            measured.append("above_floor")
        return {"name": name, "dims": dims, "measured": measured}
    if t != "tv" and abs(dims["height"] - h_spec) > 0.35 * h_spec:
        # the marked box also caught a wall, lamp or window behind the piece: report the typical height
        dims["height"] = h_spec
        measured.remove("height")
    if t == "tv":  # a wall-mounted screen: its measured height is where the bottom edge hangs
        dims["mounted_at"] = dims.pop("height")
        measured = ["width", "mounted_at"]
    if t == "round_dining":
        measured = ["height"]
    return {"name": name, "dims": dims, "measured": measured}
