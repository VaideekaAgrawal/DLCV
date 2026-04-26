"""
src/train_fusion_v2.py

Production training for SP+DINOv2 Fusion → LightGlue.
MegaDepth-1500 calibrated pairs + depth-based GT matches + NLL loss.
All gluefactory.models imports avoided (they hang). NLLLoss is inlined.
"""

from __future__ import annotations
import sys, os, json, time, random, argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "glue-factory"))

from gluefactory.geometry.wrappers import Camera, Pose
from gluefactory.geometry.gt_generation import gt_matches_from_pose_depth


# ── Inlined NLLLoss ──────────────────────────────────────────
def _weight_loss(log_assignment, weights):
    b, m, n = log_assignment.shape
    m -= 1; n -= 1
    loss_sc = log_assignment * weights
    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)
    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2)) / num_pos
    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLMatchingLoss(nn.Module):
    def __init__(self, bal=0.5):
        super().__init__()
        self.bal = bal

    def forward(self, log_assignment, gt_m0, gt_m1, gt_assign):
        m, n = gt_m0.size(-1), gt_m1.size(-1)
        w = torch.zeros_like(log_assignment)
        w[:, :m, :n] = gt_assign.float()
        w[:, :m, -1] = (gt_m0 == -1).float()
        w[:, -1, :n] = (gt_m1 == -1).float()
        nll_pos, nll_neg, num_pos, num_neg = _weight_loss(log_assignment, w)
        nll = self.bal * nll_pos + (1 - self.bal) * nll_neg
        return nll, {"nll_pos": nll_pos.mean().item(), "nll_neg": nll_neg.mean().item(),
                      "num_matchable": num_pos.mean().item()}


# ── Dataset ──────────────────────────────────────────────────
class MegaDepth1500Dataset(Dataset):
    """Reads pairs_calibrated.txt: img0 img1 K0(9) K1(9) R(9) t(3)."""

    def __init__(self, data_root, image_resize=640):
        self.root = Path(data_root)
        self.resize = image_resize
        self.pairs = []
        with open(self.root / "pairs_calibrated.txt") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 32:
                    continue
                self.pairs.append((
                    p[0], p[1],
                    np.array([float(x) for x in p[2:11]]).reshape(3, 3),
                    np.array([float(x) for x in p[11:20]]).reshape(3, 3),
                    np.array([float(x) for x in p[20:29]]).reshape(3, 3),
                    np.array([float(x) for x in p[29:32]]),
                ))
        print(f"  MegaDepth1500: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, p):
        img = Image.open(self.root / "images" / p).convert("RGB")
        ow, oh = img.size
        s = self.resize / max(ow, oh)
        nw, nh = max((int(ow * s) // 14) * 14, 14), max((int(oh * s) // 14) * 14, 14)
        img = img.resize((nw, nh), Image.BILINEAR)
        return T.ToTensor()(img), s, (nh, nw)

    def _load_depth(self, p):
        stem, scene = Path(p).stem, p.split("/")[0]
        with h5py.File(self.root / "depths" / scene / f"{stem}.h5", "r") as f:
            return torch.from_numpy(f["depth"][:]).float()

    def __getitem__(self, idx):
        i0, i1, K0, K1, R, t = self.pairs[idx]
        img0, s0, sz0 = self._load_img(i0)
        img1, s1, sz1 = self._load_img(i1)
        d0 = F.interpolate(self._load_depth(i0)[None, None], size=sz0, mode="nearest")[0, 0]
        d1 = F.interpolate(self._load_depth(i1)[None, None], size=sz1, mode="nearest")[0, 0]
        K0s, K1s = K0.copy(), K1.copy()
        K0s[0, :] *= s0; K0s[1, :] *= s0
        K1s[0, :] *= s1; K1s[1, :] *= s1
        cam0 = torch.tensor([sz0[1], sz0[0], K0s[0,0], K0s[1,1], K0s[0,2], K0s[1,2]], dtype=torch.float32)
        cam1 = torch.tensor([sz1[1], sz1[0], K1s[0,0], K1s[1,1], K1s[0,2], K1s[1,2]], dtype=torch.float32)
        T01 = torch.cat([torch.from_numpy(R).float().flatten(), torch.from_numpy(t).float()])
        return {"image0": img0, "image1": img1, "depth0": d0, "depth1": d1,
                "camera0": cam0, "camera1": cam1, "T_0to1": T01}


def _collate(batch):
    return {k: v.unsqueeze(0) for k, v in batch[0].items()}


def build_loaders(data_root, resize, ratio=0.8, seed=42, nw=2):
    ds = MegaDepth1500Dataset(data_root, resize)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    nt = int(len(ds) * ratio)
    tl = DataLoader(torch.utils.data.Subset(ds, idx[:nt]), batch_size=1, shuffle=True,
                    num_workers=nw, pin_memory=True, drop_last=True, collate_fn=_collate)
    vl = DataLoader(torch.utils.data.Subset(ds, idx[nt:]), batch_size=1, shuffle=False,
                    num_workers=nw, pin_memory=True, collate_fn=_collate)
    print(f"  Train: {nt} | Val: {len(ds)-nt}")
    return tl, vl


# ── GT matches ───────────────────────────────────────────────
def compute_gt(data, kp0, kp1, dev):
    gd = {
        "view0": {"camera": Camera(data["camera0"].to(dev)), "depth": data["depth0"].to(dev)},
        "view1": {"camera": Camera(data["camera1"].to(dev)), "depth": data["depth1"].to(dev)},
        "T_0to1": Pose(data["T_0to1"].to(dev)),
    }
    r = gt_matches_from_pose_depth(kp0, kp1, gd, pos_th=3.0, neg_th=5.0)
    return r["matches0"], r["matches1"], r["assignment"].float()


# ── Training ─────────────────────────────────────────────────
def train(
    output_dir="experiments/exp6_fusion_v2", n_ep1=30, n_ep2=20,
    strategy="gated", variant="vitb14_reg", proj_dim=128,
    lr_f=1e-3, lr_lg=5e-5, max_kp=1024, resize=640,
    accum=4, eval_every=5, device="cuda", amp=True, seed=42,
):
    from src.fusion_pipeline import FusionPipeline
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FUSION TRAINING | {strategy} | {variant} | P1={n_ep1} P2={n_ep2}")
    print("=" * 60)

    tl, vl = build_loaders(ROOT / "glue-factory/data/megadepth1500", resize, seed=seed)
    pipe = FusionPipeline(dinov2_variant=variant, fusion_strategy=strategy,
                          dino_proj_dim=proj_dim, max_keypoints=max_kp, device=device)
    loss_fn = NLLMatchingLoss()
    scaler = GradScaler("cuda", enabled=amp)
    hist = {"p1_train": [], "p1_val": [], "p2_train": [], "p2_val": [],
            "config": dict(strategy=strategy, variant=variant, proj_dim=proj_dim,
                           n_ep1=n_ep1, n_ep2=n_ep2, lr_f=lr_f, lr_lg=lr_lg,
                           max_kp=max_kp, resize=resize, seed=seed)}
    best = float("inf")

    def run_phase(tag, n_ep, opt, sched, hist_t, hist_v):
        nonlocal best
        for ep in range(1, n_ep + 1):
            t0 = time.time()
            tl_loss = _train_ep(pipe, tl, opt, scaler, loss_fn, accum, device, amp, tag)
            vl_loss, vi = _val_ep(pipe, vl, loss_fn, device, amp)
            sched.step()
            hist_t.append(tl_loss); hist_v.append(vl_loss)
            print(f"  {tag} E{ep:02d}/{n_ep}: train={tl_loss:.4f} val={vl_loss:.4f} "
                  f"m={vi:.0f} lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
            if vl_loss < best:
                best = vl_loss
                _save(pipe, opt, ep, tag, out / f"best_{tag.lower()}.pt")
            if ep % eval_every == 0:
                _save(pipe, opt, ep, tag, out / f"{tag.lower()}_e{ep:02d}.pt")
                json.dump(hist, open(out / "train_history.json", "w"), indent=2)

    # Phase 1
    print("\n── PHASE 1: Fusion only (LG frozen) ──")
    pipe.set_training_mode("fusion_only")
    opt = torch.optim.AdamW(pipe.get_trainable_params("fusion_only"), lr=lr_f, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep1, eta_min=1e-5)
    run_phase("P1", n_ep1, opt, sch, hist["p1_train"], hist["p1_val"])

    # Phase 2
    print("\n── PHASE 2: End-to-end (Fusion + LG) ──")
    pipe.set_training_mode("lg_finetune")
    pg = pipe.get_trainable_params("lg_finetune")
    pg[0]["lr"] = lr_f * 0.5; pg[1]["lr"] = lr_lg
    opt = torch.optim.AdamW(pg, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep2, eta_min=1e-6)
    run_phase("P2", n_ep2, opt, sch, hist["p2_train"], hist["p2_val"])

    _save(pipe, opt, n_ep2, "P2", out / "final_model.pt")
    json.dump(hist, open(out / "train_history.json", "w"), indent=2)
    print(f"\n✅ Done. Best val={best:.4f}  →  {out}")
    return hist


def _train_ep(pipe, loader, opt, scaler, loss_fn, accum, dev, amp_on, tag):
    pipe.dino_projection.train(); pipe.fusion.train()
    losses = []; opt.zero_grad()
    for step, data in enumerate(loader):
        try:
            with autocast(device_type="cuda", enabled=amp_on):
                pred = pipe({"image0": data["image0"].to(dev), "image1": data["image1"].to(dev)})
                if "log_assignment" not in pred:
                    continue
                m0, m1, assign = compute_gt(data, pred["keypoints0"], pred["keypoints1"], dev)
                nll, met = loss_fn(pred["log_assignment"], m0, m1, assign)
                loss = nll.mean() / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"] if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(nll.mean().item())
            if step % 100 == 0:
                print(f"    [{tag}] {step}/{len(loader)}: nll={nll.mean().item():.4f} match={met['num_matchable']:.0f}")
        except Exception as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            if step < 3:
                print(f"    [{tag}] step {step} err: {e}")
            continue
    if losses and len(losses) % accum != 0:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    return np.mean(losses) if losses else float("nan")


@torch.no_grad()
def _val_ep(pipe, loader, loss_fn, dev, amp_on):
    pipe.eval()
    losses, mc = [], []
    for data in loader:
        try:
            with autocast(device_type="cuda", enabled=amp_on):
                pred = pipe({"image0": data["image0"].to(dev), "image1": data["image1"].to(dev)})
                if "log_assignment" not in pred:
                    continue
                m0, m1, assign = compute_gt(data, pred["keypoints0"], pred["keypoints1"], dev)
                nll, _ = loss_fn(pred["log_assignment"], m0, m1, assign)
                losses.append(nll.mean().item())
                m = pred.get("matches0", torch.tensor([]))
                if m.numel() > 0:
                    mc.append((m[0] >= 0).sum().item())
        except Exception:
            continue
    pipe.dino_projection.train(); pipe.fusion.train()
    return np.mean(losses) if losses else float("nan"), np.mean(mc) if mc else 0


def _save(pipe, opt, ep, phase, path):
    torch.save({"epoch": ep, "phase": phase,
                "fusion": pipe.fusion.state_dict(),
                "dino_proj": pipe.dino_projection.state_dict(),
                "lightglue": pipe.lightglue.state_dict(),
                "optimizer": opt.state_dict()}, path)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--output_dir", default="experiments/exp6_fusion_v2")
    pa.add_argument("--strategy", default="gated", choices=["concat_mlp", "gated", "adaptive"])
    pa.add_argument("--n_ep1", type=int, default=30)
    pa.add_argument("--n_ep2", type=int, default=20)
    pa.add_argument("--lr_f", type=float, default=1e-3)
    pa.add_argument("--lr_lg", type=float, default=5e-5)
    pa.add_argument("--max_kp", type=int, default=1024)
    pa.add_argument("--resize", type=int, default=640)
    pa.add_argument("--accum", type=int, default=4)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--seed", type=int, default=42)
    a = pa.parse_args()
    train(a.output_dir, a.n_ep1, a.n_ep2, a.strategy, "vitb14_reg", 128,
          a.lr_f, a.lr_lg, a.max_kp, a.resize, a.accum, device=a.device, seed=a.seed)
