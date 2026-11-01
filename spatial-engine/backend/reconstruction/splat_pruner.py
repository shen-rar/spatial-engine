"""
Importance pruning for trained splats (in the spirit of LightGaussian, Fan et al. 2023).

A gaussian's importance is the total alpha it adds to the training images: the gradient of the
summed rendered alpha with respect to its opacity, times its opacity, summed over all views.
Faint, hidden or redundant gaussians score near zero; dropping them costs little quality but
removes their whole fill cost, which is what limits frame rate in the browser (every gaussian is
a blended quad, and laptop GPUs are fill bound).
"""
import math

import numpy as np
import torch

from backend.reconstruction.splat_trainer import ssim

PLY_KEYS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(9)],
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def read_ply(path):
    """3DGS .ply (SH degree 1) -> (header names, float32 array [n, len(names)])."""
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    names = [ln.split()[-1] for ln in raw[:end].decode().splitlines() if ln.startswith("property")]
    return names, np.frombuffer(raw[end:], dtype="<f4").reshape(-1, len(names)).copy()


def write_ply(path, names, data):
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(data.astype("<f4").tobytes())


def transform_gaussians(names, data, T):
    """Apply a similarity transform T (4x4: s*R, t) to PLY rows: centres, orientations, scales
    and the degree-1 SH colour, whose three coefficients per channel rotate like a vector."""
    col = {k: i for i, k in enumerate(names)}
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    R = T[:3, :3] / s
    out = data.copy()
    xyz = data[:, [col["x"], col["y"], col["z"]]]
    out[:, [col["x"], col["y"], col["z"]]] = xyz @ (s * R).T + T[:3, 3]
    for i in range(3):
        out[:, col[f"scale_{i}"]] += np.log(s)
    from scipy.spatial.transform import Rotation
    q = data[:, [col["rot_1"], col["rot_2"], col["rot_3"], col["rot_0"]]]  # PLY wxyz -> scipy xyzw
    q = (Rotation.from_matrix(R) * Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))).as_quat()
    out[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]] = q[:, [3, 0, 1, 2]]
    # 3DGS band 1 is SH_C1 * (-y*c0 + z*c1 - x*c2): the vector (-c2, -c0, c1) in xyz rotates by R
    for ch in range(3):
        c0, c1, c2 = (data[:, col[f"f_rest_{ch * 3 + k}"]] for k in range(3))
        v = np.stack([-c2, -c0, c1], 1) @ R.T
        out[:, col[f"f_rest_{ch * 3}"]], out[:, col[f"f_rest_{ch * 3 + 1}"]], out[:, col[f"f_rest_{ch * 3 + 2}"]] = \
            -v[:, 1], v[:, 2], -v[:, 0]
    return out


def to_gaussians(names, data, device="cuda"):
    """Renderable tensors from the PLY columns (f_rest is channel-major, as the trainers write it)."""
    col = {k: torch.tensor(data[:, i], device=device) for i, k in enumerate(names)}
    g = lambda *k: torch.stack([col[x] for x in k], 1)  # noqa: E731
    return {
        "means": g("x", "y", "z"),
        "quats": torch.nn.functional.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
        "scales": torch.exp(g("scale_0", "scale_1", "scale_2")),
        "opacities": torch.sigmoid(col["opacity"]),
        "colors": torch.cat([g("f_dc_0", "f_dc_1", "f_dc_2")[:, None],
                             g(*[f"f_rest_{i}" for i in range(9)]).reshape(-1, 3, 3).permute(0, 2, 1)], 1),
    }


def importance(gs, views, K, rasterization, W, H):
    """Summed alpha contribution of every gaussian over the given views."""
    Km = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    op = gs["opacities"].clone().requires_grad_(True)
    score = torch.zeros_like(op)
    for v in views:
        _, alpha, _ = rasterization(gs["means"], gs["quats"], gs["scales"], op, gs["colors"],
                                    v["w2c"].cuda()[None], Km, W, H, sh_degree=1)
        (g,) = torch.autograd.grad(alpha.sum(), op)
        score += g * op.detach()
    return score


def finetune(names, data, views, K, rasterization, W, H, iters=5000, log=print):
    """Recover quality after pruning: optimise the kept gaussians (no densification) with the
    trainer's loss, so neighbours take over what the dropped ones drew. Returns new PLY rows."""
    col = {k: i for i, k in enumerate(names)}
    t = lambda *k: torch.tensor(data[:, [col[x] for x in k]], device="cuda")  # noqa: E731
    p = {
        "means": t("x", "y", "z"), "scales": t("scale_0", "scale_1", "scale_2"),
        "quats": t("rot_0", "rot_1", "rot_2", "rot_3"), "opacities": t("opacity")[:, 0],
        "sh0": t("f_dc_0", "f_dc_1", "f_dc_2")[:, None],
        "shN": t(*[f"f_rest_{i}" for i in range(9)]).reshape(-1, 3, 3).permute(0, 2, 1).contiguous(),
    }
    p = {k: torch.nn.Parameter(v) for k, v in p.items()}
    centers = np.stack([np.linalg.inv(v["w2c"].numpy())[:3, 3] for v in views])
    extent = 1.1 * float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    lrs = {"means": 1.6e-6 * extent, "scales": 2e-3, "quats": 5e-4, "opacities": 1e-2, "sh0": 1e-3, "shN": 1e-3 / 20}
    opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()], eps=1e-15)
    expo = torch.zeros(len(views), 2, 3, device="cuda", requires_grad=True)
    opt_e = torch.optim.Adam([expo], lr=1e-3)
    Km = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    rng = np.random.default_rng(0)
    for it in range(1, iters + 1):
        vi = int(rng.integers(len(views)))
        img, _, _ = rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                  torch.cat([p["sh0"], p["shN"]], 1), views[vi]["w2c"].cuda()[None], Km, W, H,
                                  sh_degree=1, backgrounds=torch.rand(1, 3, device="cuda"), packed=False)
        img = img[0] * torch.exp(expo[vi, 0]) + expo[vi, 1]
        gt = views[vi]["img"].cuda(non_blocking=True)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        opt_e.step(); opt_e.zero_grad(set_to_none=True)
        if it % 1000 == 0:
            log(f"  finetune {it}/{iters} loss {loss.item():.4f}")
    out = data.copy()
    with torch.no_grad():
        rows = {"x": p["means"][:, 0], "y": p["means"][:, 1], "z": p["means"][:, 2], "opacity": p["opacities"]}
        for i in range(3):
            rows[f"scale_{i}"] = p["scales"][:, i]
            rows[f"f_dc_{i}"] = p["sh0"][:, 0, i]
        q = torch.nn.functional.normalize(p["quats"], dim=-1)
        for i in range(4):
            rows[f"rot_{i}"] = q[:, i]
        rest = p["shN"].permute(0, 2, 1).reshape(-1, 9)  # back to channel-major
        for i in range(9):
            rows[f"f_rest_{i}"] = rest[:, i]
        for k, v in rows.items():
            out[:, col[k]] = v.cpu().numpy()
    return out


def evaluate(gs, views, K, rasterization, W, H, keep=None):
    """Mean PSNR and SSIM over views, optionally rendering only the gaussians in `keep`."""
    Km = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    sel = {k: (t if keep is None else t[keep]) for k, t in gs.items()}
    ps, ss = [], []
    with torch.no_grad():
        for v in views:
            im, _, _ = rasterization(sel["means"], sel["quats"], sel["scales"], sel["opacities"], sel["colors"],
                                     v["w2c"].cuda()[None], Km, W, H, sh_degree=1)
            im, gt = im[0].clamp(0, 1), v["img"].cuda()
            ps.append(-10 * math.log10(max(((im - gt) ** 2).mean().item(), 1e-8)))
            ss.append(ssim(im, gt).item())
    return float(np.mean(ps)), float(np.mean(ss))
