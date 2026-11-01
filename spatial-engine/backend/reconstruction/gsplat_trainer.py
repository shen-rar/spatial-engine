"""
Full-resolution Gaussian Splatting with gsplat's CUDA rasteriser (Nerfstudio, Apache-2.0).

Same inputs and output as splat_trainer.train (views and intrinsics from load_colmap_views, SfM
points plus fill seeds, a .ply with SH degree 1 in the gravity-aligned frame), so the viewer,
the .ksplat converter and the blueprint builder need no changes. gsplat is fast enough to train on
the full keyframes (1600 px) for 30k steps, where the pure PyTorch trainer is limited to ~480 px.

  * MCMC densification (Kheradmand et al. 2024): a fixed gaussian budget, relocated from dead
    gaussians to where the images need detail; no opacity resets, fewer floaters
  * L1 + D-SSIM loss, opacity and scale regularisers, per-view exposure gain/bias
  * random background each step, so holes cannot hide behind a fixed colour

gsplat compiles its CUDA kernels on first use: it needs the CUDA toolkit matching torch (12.8 for
an RTX 50-series GPU) and, on Windows, the MSVC C++ build tools.
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from backend.reconstruction.splat_trainer import SH_C0, fill_seeds, ssim


def _import_gsplat():
    """Import gsplat, which JIT-compiles its CUDA kernels on first use. On Windows every source
    gets win_cuda_fix.h pre-included (the `small` macro from windows.h breaks torch's headers)
    and gsplat's GCC-style C++ flags become MSVC ones; this runs on every import, as torch
    re-checks the build against its flags."""
    if os.name == "nt":
        import torch.utils.cpp_extension as ext
        if not getattr(ext._jit_compile, "win_cuda_fix", False):
            orig, hdr = ext._jit_compile, str(Path(__file__).with_name("win_cuda_fix.h"))

            def jit(name, sources, extra_cflags, extra_cuda_cflags, *a, **k):
                cflags = ["/O2" if f.startswith("-O") else f for f in extra_cflags or [] if not f.startswith("-W")]
                return orig(name, sources, cflags + [f"/FI{hdr}"],
                            list(extra_cuda_cflags or []) + ["--pre-include", hdr], *a, **k)
            jit.win_cuda_fix = True
            ext._jit_compile = jit
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    return rasterization, MCMCStrategy


def train(views, K, xyz, rgb, out_path: Path, iters=30000, device="cuda", log=print, cap=500_000,
          fill=True, holdout=True):
    from scipy.spatial import cKDTree

    rasterization, MCMCStrategy = _import_gsplat()

    dev = torch.device(device)
    H, W = views[0]["img"].shape[:2]
    fx, fy, cx, cy = K
    Km = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=dev)
    rng = np.random.default_rng(0)
    xyz, rgb = np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64)
    hold = list(range(0, len(views), 8)) if holdout else []
    tr = [i for i in range(len(views)) if i not in hold]
    n_sfm = len(xyz)
    if fill:
        # same seed density per image area as the 480 px trainer
        seeds, scol = fill_seeds([views[i] for i in tr], K, xyz, rgb, step=max(12, round(12 * W / 480)),
                                 gap_px=10.0 * W / 480)
        xyz, rgb = np.concatenate([xyz, seeds]), np.concatenate([rgb, scol])
        log(f"fill seeds {len(seeds)} added to {n_sfm} SfM points")

    dist, _ = cKDTree(xyz).query(xyz, k=4)  # first neighbour is the point itself
    knn = np.clip(dist[:, 1:].mean(1), 1e-3, None)
    n = len(xyz)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(xyz)),
        "scales": torch.nn.Parameter(t(np.log(knn))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((t(rgb) / 255.0 - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, 3, 3, device=dev)),  # SH degree 1: [basis, rgb]
    })

    centers = np.stack([np.linalg.inv(v["w2c"].numpy())[:3, 3] for v in views])
    extent = 1.1 * float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_decay = 0.01 ** (1.0 / iters)  # means lr falls 100x over the run
    expo = torch.zeros(len(views), 2, 3, device=dev, requires_grad=True)  # per-view gain (log), bias
    opt_expo = torch.optim.Adam([expo], lr=1e-3)

    strategy = MCMCStrategy(cap_max=max(cap, n), refine_stop_iter=int(iters * 0.8))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    t0 = time.time()
    for it in range(1, iters + 1):
        vi = tr[rng.integers(len(tr))]
        vw = views[vi]
        bg = torch.rand(1, 3, device=dev)
        img, _, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.cat([params["sh0"], params["shN"]], 1), vw["w2c"].to(dev)[None], Km[None], W, H,
            sh_degree=min(1, it // 1000), backgrounds=bg, packed=False)
        img = img[0] * torch.exp(expo[vi, 0]) + expo[vi, 1]
        gt = vw["img"].to(dev, non_blocking=True)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt))
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        strategy.step_pre_backward(params, optimizers, state, it, info)
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opt_expo.step()
        opt_expo.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] *= means_decay
        strategy.step_post_backward(params, optimizers, state, it, info,
                                    lr=optimizers["means"].param_groups[0]["lr"])
        if it % 500 == 0:
            log(f"it {it}/{iters} loss {loss.item():.4f} gaussians {len(params['means'])} {time.time() - t0:.0f}s")

    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) > 0.005
        ps, ss = [], []
        for i in hold:
            im, _, _ = rasterization(
                params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                torch.cat([params["sh0"], params["shN"]], 1), views[i]["w2c"].to(dev)[None], Km[None], W, H,
                sh_degree=1)
            im, gt = im[0].clamp(0, 1), views[i]["img"].to(dev)
            ps.append(-10 * math.log10(max(((im - gt) ** 2).mean().item(), 1e-8)))
            ss.append(ssim(im, gt).item())
        psnr = float(np.mean(ps)) if ps else float("nan")
        if ps:
            log(f"held-out PSNR {psnr:.2f} dB  SSIM {np.mean(ss):.3f} over {len(ps)} views")
        p = {k: v.detach()[keep] for k, v in params.items()}
    log(f"final gaussians {int(keep.sum())}")
    _save_ply(p, out_path)
    return psnr


def _save_ply(p, path: Path):
    """Write the 3DGS .ply layout splat_trainer.save_ply uses (f_rest channel-major)."""
    n = len(p["means"])
    cols = [p["means"], torch.zeros(n, 3, device=p["means"].device), p["sh0"][:, 0],
            p["shN"].permute(0, 2, 1).reshape(n, 9), p["opacities"][:, None], p["scales"],
            torch.nn.functional.normalize(p["quats"], dim=-1)]
    data = torch.cat(cols, 1).cpu().numpy().astype("<f4")
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(9)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(data.tobytes())
