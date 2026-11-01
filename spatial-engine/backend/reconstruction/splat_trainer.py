"""
Compact 3D Gaussian Splatting trainer in pure PyTorch (no CUDA extensions to compile).

Tile-based differentiable rasteriser: per-tile gaussians are depth sorted and alpha composited.
Colour is SH degree 1 (view-dependent, for glossy floors and glass). Trains directly in the
gravity-aligned metric frame produced by MetricGravityAligner so the .ply drops into the viewer
next to points.ply / collision_mesh.glb. Slower than the reference CUDA implementation; it runs
anywhere torch does.

Quality measures beyond plain 3DGS-on-SfM-points:
  * L1 + D-SSIM loss, screen-space-gradient densification, periodic opacity reset (kills floaters)
  * per-view exposure gain/bias (phone video auto-exposure would otherwise become floaters)
  * random background each step, so holes cannot hide behind a black background
  * fill seeds: gaussians guessed where the sparse SfM cloud leaves gaps in a view, placed at
    depth interpolated from neighbouring SfM points or on the floor plane (Y = 0)
  * world-size cap and anisotropy penalty, so no room-sized blobs or needles smear across views
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

TILE = 16
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
CHUNK_ELEMS = 1 << 23 # tiles x pixels-per-tile x gaussians-per-tile per shading chunk


def load_colmap_views(rec, image_dir: Path, sim3: np.ndarray, width: int):
    """Undistort + downscale images (width 0 keeps their size); return views (image, w2c in aligned frame) and intrinsics."""
    S = sim3[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(S))))
    Rg = S / s
    views = []
    K = None
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        k1 = cam.params[3] if len(cam.params) > 3 else 0.0
        raw = cv2.imread(str(image_dir / img.name))
        if raw is None:
            continue
        Km = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        und = cv2.undistort(raw, Km, np.array([k1, 0, 0, 0], dtype=np.float64))
        scale = width / und.shape[1] if width else 1.0  # width 0: keep the keyframe resolution
        if scale != 1.0:
            und = cv2.resize(und, (width, int(round(und.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        K = (f * scale, f * scale, cx * scale, cy * scale)
        c2w = img.cam_from_world().inverse().matrix()  # 3x4, colmap frame
        R_c2w = Rg @ c2w[:, :3]
        center = S @ c2w[:, 3] + sim3[:3, 3]
        w2c = np.eye(4)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -R_c2w.T @ center
        views.append({
            "img": torch.from_numpy(cv2.cvtColor(und, cv2.COLOR_BGR2RGB)).float() / 255.0,
            "w2c": torch.from_numpy(w2c).float(),
        })
    return views, K


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


PARAMS = ("means", "log_scales", "quats", "opac", "colors", "sh1")


class Splats(torch.nn.Module):
    def __init__(self, xyz, rgb, init_scale, init_opac=None):
        super().__init__()
        n = len(xyz)
        self.means = torch.nn.Parameter(xyz.clone())
        self.log_scales = torch.nn.Parameter(torch.log(init_scale.clamp(min=1e-3))[:, None].repeat(1, 3))
        q = torch.zeros(n, 4, device=xyz.device)
        q[:, 0] = 1
        self.quats = torch.nn.Parameter(q)
        op = torch.full((n,), -1.0, device=xyz.device) if init_opac is None else init_opac
        self.opac = torch.nn.Parameter(op)
        self.colors = torch.nn.Parameter(torch.logit(rgb.clamp(0.02, 0.98)))
        self.sh1 = torch.nn.Parameter(torch.zeros(n, 3, 3, device=xyz.device))  # [basis, rgb]

    def n(self):
        return self.means.shape[0]


def render(sp: Splats, w2c, K, H, W, bg=0.0, max_per_tile=4096, screen=None, values=None):
    """screen: optional zeros [n, 2] requiring grad; its .grad is the screen-space gradient.
    values: optional [n, 3] per-gaussian values composited instead of colour (e.g. depth)."""
    fx, fy, cx, cy = K
    dev = sp.means.device
    p_cam = sp.means @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    zc = z.clamp(min=0.05)
    u = fx * p_cam[:, 0] / zc + cx
    v = fy * p_cam[:, 1] / zc + cy
    if screen is not None:
        u, v = u + screen[:, 0], v + screen[:, 1]
    R = quat_to_rot(sp.quats)
    S = torch.exp(sp.log_scales)
    M = R * S[:, None, :]
    cov3 = M @ M.transpose(1, 2)
    Rw = w2c[:3, :3]
    cov_cam = Rw @ cov3 @ Rw.T
    J = torch.zeros(sp.n(), 2, 3, device=dev)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * p_cam[:, 0] / zc ** 2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * p_cam[:, 1] / zc ** 2
    cov2 = J @ cov_cam @ J.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp(min=1e-6)
    ia, ib, ic = c / det, -b / det, a / det
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    rad = 3.0 * torch.sqrt(lam)
    opac = torch.sigmoid(sp.opac)
    # SH degree 1, same convention as the reference 3DGS / web viewers
    campos = -w2c[:3, :3].T @ w2c[:3, 3]
    d = torch.nn.functional.normalize(sp.means - campos, dim=-1)
    sh = sp.sh1
    col = torch.sigmoid(sp.colors) + SH_C1 * (-d[:, 1:2] * sh[:, 0] + d[:, 2:3] * sh[:, 1] - d[:, 0:1] * sh[:, 2])
    col = col.clamp(0, 1) if values is None else values

    valid = (z > 0.05) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H) & (rad < 0.6 * W)
    order = torch.argsort(torch.where(valid, z, torch.full_like(z, 1e9)))[: int(valid.sum())]
    u, v, rad = u[order], v[order], rad[order]
    ia, ib, ic, opac, col = ia[order], ib[order], ic[order], opac[order], col[order]

    ty, tx = math.ceil(H / TILE), math.ceil(W / TILE)
    bg = torch.as_tensor(bg, dtype=torch.float32, device=dev).expand(3)
    n = u.shape[0]
    if n == 0:
        return bg.expand(H, W, 3).clone()
    x0 = torch.arange(tx, device=dev) * TILE
    y0 = torch.arange(ty, device=dev) * TILE
    ov_x = (u[None, :] + rad[None, :] >= x0[:, None]) & (u[None, :] - rad[None, :] <= (x0[:, None] + TILE))
    ov_y = (v[None, :] + rad[None, :] >= y0[:, None]) & (v[None, :] - rad[None, :] <= (y0[:, None] + TILE))
    counts = (ov_y.float() @ ov_x.float().T).reshape(-1)  # gaussians per tile

    # All tiles in batched passes instead of a Python loop per tile. Tiles are grouped by load,
    # so a chunk pads only to tiles of similar size; gaussians are depth sorted, so a stable sort
    # of "not in tile" keeps each tile's first k hits front to back.
    py, px = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    py, px = py.reshape(-1).float() + 0.5, px.reshape(-1).float() + 0.5
    tile_y = y0[:, None].expand(ty, tx).reshape(-1).float()
    tile_x = x0[None, :].expand(ty, tx).reshape(-1).float()
    by_load = torch.argsort(counts, descending=True)
    loads = counts[by_load].clamp(1, max_per_tile).long().tolist()
    out = torch.empty(ty * tx, TILE * TILE, 3, device=dev)
    s = 0
    while s < len(loads):
        k = loads[s]
        e = min(len(loads), s + max(1, CHUNK_ELEMS // (k * TILE * TILE)))
        tj = by_load[s:e]
        mask = ov_y[tj // tx] & ov_x[tj % tx]                                    # [t, n]
        idx = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)[:, :k]  # [t, k]
        hit = mask.gather(1, idx).float()
        X = tile_x[tj, None] + px[None, :]                                         # [t, P]
        Y = tile_y[tj, None] + py[None, :]
        args = (u, v, ia, ib, ic, opac, col, idx, hit, X, Y, bg)
        if torch.is_grad_enabled():
            # recompute each chunk in backward instead of keeping every [t, P, k] intermediate:
            # memory stays bounded by one chunk, which avoids spilling out of GPU memory
            out[tj] = checkpoint(_shade_tiles, *args, use_reentrant=False)
        else:
            out[tj] = _shade_tiles(*args)
        s = e
    img = out.reshape(ty, tx, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(ty * TILE, tx * TILE, 3)
    return img[:H, :W]


def _shade_tiles(u, v, ia, ib, ic, opac, col, idx, hit, X, Y, bg):
    """Alpha-composite the depth-sorted gaussians idx [t, k] over pixel centres X, Y [t, P]."""
    dx = X[:, :, None] - u[idx][:, None, :]                                       # [t, P, k]
    dy = Y[:, :, None] - v[idx][:, None, :]
    pw = -0.5 * (ia[idx][:, None] * dx * dx + 2 * ib[idx][:, None] * dx * dy + ic[idx][:, None] * dy * dy)
    alpha = (opac[idx][:, None] * torch.exp(pw.clamp(max=0))).clamp(max=0.99) * hit[:, None]
    alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1]], -1), -1)
    wgt = alpha * T
    return torch.einsum("tpk,tkc->tpc", wgt, col[idx]) + (1 - wgt.sum(-1, keepdim=True)) * bg


def save_ply(sp: Splats, path: Path):
    m = sp.means.detach().cpu().numpy()
    dc = ((torch.sigmoid(sp.colors) - 0.5) / SH_C0).detach().cpu().numpy()
    # f_rest is channel-major: all R coefficients, then G, then B
    rest = sp.sh1.detach().permute(0, 2, 1).reshape(-1, 9).cpu().numpy()
    op = sp.opac.detach().cpu().numpy()[:, None]
    sc = sp.log_scales.detach().cpu().numpy()
    q = torch.nn.functional.normalize(sp.quats, dim=-1).detach().cpu().numpy()
    n = len(m)
    data = np.concatenate([m, np.zeros((n, 3)), dc, rest, op, sc, q], 1).astype("<f4")
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(9)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(data.tobytes())


def load_ply(path: Path, device="cpu") -> Splats:
    """Read a .ply written by save_ply (SH degree 0 or 1) back into Splats."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    names = [ln.split()[-1] for ln in raw[:end].decode().splitlines() if ln.startswith("property")]
    d = np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(names))
    c = {k: d[:, i] for i, k in enumerate(names)}
    sp = Splats(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1))
    P = lambda x: torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))  # noqa: E731
    sp.means = P(np.stack([c["x"], c["y"], c["z"]], 1))
    sp.log_scales = P(np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], 1))
    sp.quats = P(np.stack([c["rot_0"], c["rot_1"], c["rot_2"], c["rot_3"]], 1))
    sp.opac = P(c["opacity"])
    rgb = np.clip(np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], 1) * SH_C0 + 0.5, 1e-4, 1 - 1e-4)
    sp.colors = P(np.log(rgb / (1 - rgb)))
    rest = np.stack([c[f"f_rest_{i}"] for i in range(9)], 1) if "f_rest_0" in c else np.zeros((len(d), 9))
    sp.sh1 = P(rest.reshape(-1, 3, 3).transpose(0, 2, 1))
    return sp.to(device)


def ssim(x, y):
    """Mean SSIM of two [H, W, 3] images (11x11 gaussian window)."""
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    g = torch.tensor(cv2.getGaussianKernel(11, 1.5), dtype=torch.float32, device=x.device)
    w = (g @ g.T)[None, None].repeat(3, 1, 1, 1)
    f = lambda t: torch.nn.functional.conv2d(t, w, padding=5, groups=3)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean()


def fill_seeds(views, K, xyz, rgb, step=12, gap_px=10.0, max_seeds=30000):
    """Guess gaussians for image regions the sparse SfM cloud leaves empty.

    For each view, SfM points are projected; grid pixels farther than gap_px from any projected
    point get a seed on their viewing ray, at the median depth of the 8 nearest projected points,
    or on the floor plane (Y = 0) when the ray hits the floor first. Colour is the pixel colour.
    Seeds start fairly transparent: the optimiser keeps the ones the other views agree with.
    """
    from scipy.spatial import cKDTree

    fx, fy, cx, cy = K
    H, W = views[0]["img"].shape[:2]
    gy, gx = np.mgrid[step // 2:H:step, step // 2:W:step]
    gx, gy = gx.reshape(-1).astype(np.float64), gy.reshape(-1).astype(np.float64)
    seeds, cols = [], []
    for vw in views:
        w2c = vw["w2c"].numpy().astype(np.float64)
        pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        ok = pc[:, 2] > 0.2
        uv = np.stack([fx * pc[ok, 0] / pc[ok, 2] + cx, fy * pc[ok, 1] / pc[ok, 2] + cy], 1)
        inside = (uv[:, 0] > -W * 0.25) & (uv[:, 0] < W * 1.25) & (uv[:, 1] > -H * 0.25) & (uv[:, 1] < H * 1.25)
        uv, depth = uv[inside], pc[ok, 2][inside]
        if len(uv) < 20:
            continue
        tree = cKDTree(uv)
        dist, nn = tree.query(np.stack([gx, gy], 1), k=8)
        gap = dist[:, 0] > gap_px
        if not gap.any():
            continue
        z = np.median(depth[nn[gap]], axis=1)
        rays_cam = np.stack([(gx[gap] - cx) / fx, (gy[gap] - cy) / fy, np.ones(gap.sum())], 1)
        R_c2w = w2c[:3, :3].T
        center = -R_c2w @ w2c[:3, 3]
        rays = rays_cam @ R_c2w.T
        # floor plane Y = 0 (gravity-aligned frame): ray depth where it meets the floor
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = np.where(rays[:, 1] < -1e-3, -center[1] / rays[:, 1], np.inf)
        z = np.minimum(z, t_floor)
        seeds.append(center + rays * z[:, None])
        img = vw["img"].numpy()
        cols.append(img[gy[gap].astype(int), gx[gap].astype(int)] * 255)
    if not seeds:
        return np.zeros((0, 3)), np.zeros((0, 3))
    seeds, cols = np.concatenate(seeds), np.concatenate(cols)
    if len(seeds) > max_seeds:
        pick = np.random.default_rng(0).choice(len(seeds), max_seeds, replace=False)
        seeds, cols = seeds[pick], cols[pick]
    return seeds, cols


def _rebuild(sp, opt, src, overrides):
    """Replace every parameter by p[src] (plus overrides[name], a full new tensor) and carry the
    Adam moments along, so densification does not reset the optimiser."""
    for gi, name in enumerate(PARAMS):
        old = getattr(sp, name)
        new = torch.nn.Parameter((overrides[name] if name in overrides else old.detach()[src]).contiguous())
        st = opt.state.pop(old, None)
        if st:
            st["exp_avg"] = st["exp_avg"][src].contiguous()
            st["exp_avg_sq"] = st["exp_avg_sq"][src].contiguous()
            opt.state[new] = st
        opt.param_groups[gi]["params"] = [new]
        setattr(sp, name, new)


def _densify(sp, opt, gacc, gcnt, grad_thresh, size_thresh, room):
    """Clone small / split large gaussians whose mean screen-space gradient exceeds grad_thresh
    (at most `room` of them, highest gradient first), as in 3DGS. Transparent and oversized
    gaussians are pruned. size_thresh is relative to the scene, so large rooms densify too."""
    avg = gacc / gcnt.clamp(min=1)
    sel = avg > grad_thresh
    if int(sel.sum()) > room:
        sel = torch.zeros_like(sel)
        sel[torch.topk(avg, max(room, 0)).indices] = True
    with torch.no_grad():
        scales = torch.exp(sp.log_scales)
        big = scales.max(1).values > size_thresh
        clone, split = sel & ~big, sel & big
        oversized = scales.max(1).values > 10 * size_thresh
        keep = (torch.sigmoid(sp.opac) > 0.005) & ~split & ~oversized
        i_keep, i_clone, i_split = keep.nonzero()[:, 0], clone.nonzero()[:, 0], split.nonzero()[:, 0]
        src = torch.cat([i_keep, i_clone, i_split, i_split])
        means = sp.means.detach()
        rot = quat_to_rot(sp.quats.detach()[i_split])
        kids = [means[i_split] + (rot @ (torch.randn_like(scales[i_split]) * scales[i_split])[..., None]).squeeze(-1)
                for _ in range(2)]
        ls = sp.log_scales.detach()
        _rebuild(sp, opt, src, {
            "means": torch.cat([means[i_keep], means[i_clone], *kids]),
            "log_scales": torch.cat([ls[i_keep], ls[i_clone], ls[i_split] - math.log(1.6), ls[i_split] - math.log(1.6)]),
        })
    return int(clone.sum()), int(split.sum()), int((~keep & ~split).sum())


def train(views, K, xyz, rgb, out_path: Path, iters=2500, device="cuda", log=print, max_gauss=250000,
          fill=True, holdout=True):
    dev = torch.device(device)
    H, W = views[0]["img"].shape[:2]
    rng = np.random.default_rng(0)
    xyz, rgb = np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64)
    hold = list(range(0, len(views), 8)) if holdout else []
    tr = [i for i in range(len(views)) if i not in hold]
    n_sfm = len(xyz)
    init_opac = np.full(n_sfm, -1.0)
    if fill:
        seeds, scol = fill_seeds([views[i] for i in tr], K, xyz, rgb)
        xyz, rgb = np.concatenate([xyz, seeds]), np.concatenate([rgb, scol])
        init_opac = np.concatenate([init_opac, np.full(len(seeds), -2.0)])
        log(f"fill seeds {len(seeds)} added to {n_sfm} SfM points")
    pts = torch.from_numpy(xyz).float().to(dev)
    cols = torch.from_numpy(rgb).float().to(dev) / 255.0
    # initial scale = mean distance to 3 nearest neighbours; a KD-tree avoids an N x N matrix
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(xyz).query(xyz, k=4)  # first neighbour is the point itself
    knn = torch.from_numpy(dist[:, 1:].mean(1)).float().to(dev)
    sp = Splats(pts, cols, knn, torch.from_numpy(init_opac).float().to(dev)).to(dev)
    size_thresh = 0.01 * float(np.linalg.norm(np.ptp(xyz[:n_sfm], axis=0)))  # 1% of the scene diagonal
    max_scale = math.log(5 * size_thresh)

    centers = np.stack([np.linalg.inv(v["w2c"].numpy())[:3, 3] for v in views])
    extent = 1.1 * float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    lr_means0, lr_means1 = 1.6e-4 * extent, 1.6e-6 * extent
    lrs = {"means": lr_means0, "log_scales": 5e-3, "quats": 1e-3, "opac": 5e-2, "colors": 2.5e-2,
           "sh1": 2.5e-3}
    opt = torch.optim.Adam([{"params": [getattr(sp, k)], "lr": lrs[k]} for k in PARAMS], eps=1e-15)
    # per-view exposure: gain (log) and bias; folded away when exporting
    expo = torch.zeros(len(views), 2, 3, device=dev, requires_grad=True)
    opt_expo = torch.optim.Adam([expo], lr=1e-3)

    densify_until = int(iters * 0.6)
    reset_every = max(1000, iters // 5)
    grad_thresh = 1e-4 * 2.0 / W  # half the 3DGS NDC threshold (in pixels): sparse SfM needs more growth
    gacc = torch.zeros(sp.n(), device=dev)
    gcnt = torch.zeros(sp.n(), device=dev)
    t0 = time.time()
    for it in range(1, iters + 1):
        frac = (it - 1) / max(1, iters - 1)
        opt.param_groups[0]["lr"] = math.exp((1 - frac) * math.log(lr_means0) + frac * math.log(lr_means1))
        vi = tr[rng.integers(len(tr))]
        vw = views[vi]
        screen = torch.zeros(sp.n(), 2, device=dev, requires_grad=True)
        bg = torch.rand(3, device=dev)
        img = render(sp, vw["w2c"].to(dev), K, H, W, bg=bg, screen=screen)
        img = img * torch.exp(expo[vi, 0]) + expo[vi, 1]
        gt = vw["img"].to(dev)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt))
        sc = torch.exp(sp.log_scales)
        loss = loss + 0.01 * torch.relu(sc.max(1).values / sc.min(1).values.clamp(min=1e-6) - 10).mean()
        opt.zero_grad(set_to_none=True)
        opt_expo.zero_grad(set_to_none=True)
        loss.backward()
        if it < densify_until and screen.grad is not None:
            g = screen.grad.norm(dim=-1)
            gacc += g
            gcnt += (g > 0).float()
        opt.step()
        opt_expo.step()
        with torch.no_grad():
            sp.log_scales.clamp_(max=max_scale)
        if 300 <= it < densify_until and it % 100 == 0:
            room = max_gauss - sp.n()
            nc, ns, npr = _densify(sp, opt, gacc, gcnt, grad_thresh, size_thresh, room)
            gacc = torch.zeros(sp.n(), device=dev)
            gcnt = torch.zeros(sp.n(), device=dev)
            if it % 500 == 0:
                log(f"  densify: +{nc} clone +{ns} split -{npr} pruned")
        if it % reset_every == 0 and it < densify_until - 200:
            # opacity reset: floaters that no view really needs fade out and get pruned
            with torch.no_grad():
                sp.opac.clamp_(max=math.log(0.01 / 0.99))
            st = opt.state.get(sp.opac)
            if st:
                st["exp_avg"].zero_()
                st["exp_avg_sq"].zero_()
        if it % 100 == 0:
            log(f"it {it}/{iters} loss {loss.item():.4f} gaussians {sp.n()} {time.time() - t0:.0f}s")

    # final prune of near-invisible gaussians
    with torch.no_grad():
        keep = (torch.sigmoid(sp.opac) > 0.005).nonzero()[:, 0]
        _rebuild(sp, opt, keep, {})
    with torch.no_grad():
        ps, ss = [], []
        for i in hold:
            im = render(sp, views[i]["w2c"].to(dev), K, H, W).clamp(0, 1)
            gt = views[i]["img"].to(dev)
            ps.append(-10 * math.log10(max(((im - gt) ** 2).mean().item(), 1e-8)))
            ss.append(ssim(im, gt).item())
    psnr = float(np.mean(ps)) if ps else float("nan")
    if ps:
        log(f"held-out PSNR {psnr:.2f} dB  SSIM {np.mean(ss):.3f} over {len(ps)} views")
    log(f"final gaussians {sp.n()}")
    save_ply(sp, out_path)
    return psnr
