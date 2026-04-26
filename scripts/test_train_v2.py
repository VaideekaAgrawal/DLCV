#!/usr/bin/env python
"""Quick smoke test for train_fusion_v2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_fusion_v2 import NLLMatchingLoss, MegaDepth1500Dataset
import torch

print("✓ imports ok")

loss = NLLMatchingLoss()
la = torch.randn(1, 11, 11)
m0 = torch.tensor([[1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
m1 = torch.tensor([[-1, 0, -1, -1, -1, -1, -1, -1, -1, -1]])
a = torch.zeros(1, 10, 10); a[0, 0, 1] = 1
nll, met = loss(la, m0, m1, a)
print(f"✓ NLLLoss: {nll.item():.4f}, metrics={met}")

ds = MegaDepth1500Dataset("glue-factory/data/megadepth1500", 320)
s = ds[0]
print(f"✓ Dataset: img0={s['image0'].shape} depth0={s['depth0'].shape} cam0={s['camera0']}")
print(f"✓ T_0to1 shape={s['T_0to1'].shape}")

print("\n✅ ALL SMOKE TESTS PASSED")
