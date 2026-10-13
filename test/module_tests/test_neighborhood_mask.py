#!/usr/bin/env python3
"""inter_loss_mode="hotspot_neighborhood" 平行模式的单元测试 (CPU, 无模型)。

覆盖:
1. derive_hotspot_neighborhood_mask 的几何语义 — 近热点×埋藏残基被排除,
   近热点×可及残基被保留, 远热点残基被排除, 热点本体强制入选;
2. 邻域过小时回退 None (调用方回退全覆盖);
3. batch 维广播一致;
4. compute_structure_losses 兼容性 — mask 全 True 时与默认全覆盖逐位一致,
   部分 mask 正常运行; 原签名调用不变。

用法: python test/module_tests/test_neighborhood_mask.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import esmcp.binder_design as bd  # noqa: E402


def logits_from_distances(dist: torch.Tensor, bins: torch.Tensor,
                          sharpness: float = 5.0) -> torch.Tensor:
    """(T, T) 距离矩阵 → (T, T, nbins) logits, softmax 集中于最近 bin。"""
    return -sharpness * (bins[None, None, :] - dist[:, :, None]) ** 2


def build_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """10 残基合成几何: 热点=9。

    - 0-4: 埋藏簇 (两两 5Å, 互相接触 → 接触数高), 距热点 40Å;
    - 5:   距热点 7Å, 无接触 → 可及;
    - 6:   距热点 8Å 但接触簇内 3/4 (8Å) → 埋藏;
    - 7/8: 距热点 12/13Å, 无接触 → 可及;
    - 9:   热点本体。
    期望 mask = {5, 7, 8, 9} (6 被埋藏过滤排除)。
    """
    T = 10
    d = torch.full((T, T), 40.0)
    d.fill_diagonal_(0.0)
    cluster = list(range(5))
    for i in cluster:
        for j in cluster:
            if i != j:
                d[i, j] = 5.0
    d[5, 9] = d[9, 5] = 7.0
    d[6, 9] = d[9, 6] = 8.0
    d[6, 3] = d[3, 6] = 8.0
    d[6, 4] = d[4, 6] = 8.0
    d[7, 9] = d[9, 7] = 12.0
    d[8, 9] = d[9, 8] = 13.0
    return d, torch.tensor([9]), torch.tensor([5, 7, 8, 9])


def main() -> None:
    bins = bd.get_mid_points()
    d, hot, expected = build_geometry()
    logits = logits_from_distances(d, bins)

    # 1. 几何语义
    mask = bd.derive_hotspot_neighborhood_mask(
        logits[None], bins, hot, min_size=2
    )
    assert mask is not None and mask.shape == (1, 10), mask
    got = mask[0].nonzero(as_tuple=True)[0]
    assert torch.equal(got, expected), f"mask={got.tolist()}, expected={expected.tolist()}"
    print(f"[1] 几何语义 OK: mask={got.tolist()} (6 被埋藏过滤排除, 0-4 过远)")

    # 2. 邻域过小回退 None (默认 min_size=10 > 4)
    assert bd.derive_hotspot_neighborhood_mask(logits[None], bins, hot) is None
    print("[2] min_size 回退 OK: 4 < 10 → None")

    # 3. batch 广播一致
    mask_b2 = bd.derive_hotspot_neighborhood_mask(
        torch.stack([logits, logits]), bins, hot, min_size=2
    )
    assert mask_b2 is not None and mask_b2.shape == (2, 10)
    assert torch.equal(mask_b2[0], mask_b2[1])
    print("[3] batch=2 一致 OK")

    # 4. compute_structure_losses 兼容性
    torch.manual_seed(0)
    T_target, binder_len = 4, 3
    dgram = torch.randn(1, T_target + binder_len, T_target + binder_len, bins.numel())
    base = bd.compute_structure_losses(dgram, binder_len)  # 原签名原语义
    all_true = torch.ones(1, T_target, dtype=torch.bool)
    para = bd.compute_structure_losses(
        dgram, binder_len, inter_neighborhood_mask=all_true
    )
    assert torch.allclose(base["inter_contact_loss"], para["inter_contact_loss"]), (
        base["inter_contact_loss"], para["inter_contact_loss"]
    )
    partial = bd.compute_structure_losses(
        dgram, binder_len,
        inter_neighborhood_mask=torch.tensor([[True, False, True, True]]),
    )
    assert not torch.allclose(base["inter_contact_loss"], partial["inter_contact_loss"])
    assert torch.isfinite(partial["total_loss"]).all()
    print(f"[4] 兼容性 OK: 全True mask ≡ 全覆盖 "
          f"({base['inter_contact_loss'].item():.4f}); 部分 mask 值不同但有限 "
          f"({partial['inter_contact_loss'].item():.4f})")

    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
