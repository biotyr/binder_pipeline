#!/usr/bin/env python3
"""pair_chains_iptm 引入的单元测试 (CPU, 无模型)。

覆盖 (2026/10/19):
1. merge_pair_chains_iptm 的合并语义 — 多链残基加权 (t2b 方向
   [target][binder]), 方向不对称性 (t2b ≠ b2t), 单链退化 (merged =
   pair 本身), batch 维取第一个元素;
2. REMARK 250 写入 → _parse_remarks 回读 (标量键可解析,
   pair_chains_iptm_chains 人类可读行不进解析);
3. quality_filter 的 min_pair_chains_iptm 门 — 超阈值 PASS / 低阈值
   FAIL / REMARK 缺失 FAIL (向后兼容语义)。

用法: python test/module_tests/test_pair_chains_iptm.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import esmcp.binder_design as bd  # noqa: E402
from inverse_optimize.quality_filter import (  # noqa: E402
    QualityFilterConfig,
    _parse_remarks,
    filter_pdb_by_quality,
)


MINIMAL_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  "
    "1.00 80.00           C\n"
    "ATOM      2  CA  GLY A   2      10.000   0.000   0.000  "
    "1.00 80.00           C\n"
)


def test_merge_weighted_t2b() -> None:
    """多链残基加权: merged_t2b = Σ Lᵢ·pair[tᵢ][binder] / Σ Lᵢ, 且方向正确。"""
    # 3 链: A(100), B(122) 为 target, C 为 binder。矩阵方向敏感:
    # [A][C]=0.8 与 [C][A]=0.3 刻意不对称。float64 保证期望值精确可比。
    m = torch.zeros(3, 3, dtype=torch.float64)
    m[0, 2] = 0.8   # pair[A][C]  t2b
    m[2, 0] = 0.3   # pair[C][A]  b2t
    m[1, 2] = 0.6   # pair[B][C]  t2b
    m[2, 1] = 0.5   # pair[C][B]  b2t
    detail = bd.merge_pair_chains_iptm(m, [100, 122])

    exp_t2b = (100 * 0.8 + 122 * 0.6) / 222
    exp_b2t = (100 * 0.3 + 122 * 0.5) / 222
    assert abs(detail["merged_t2b"] - exp_t2b) < 1e-9, detail
    assert abs(detail["merged_b2t"] - exp_b2t) < 1e-9, detail
    assert detail["chains"]["A"] == {"t2b": 0.8, "b2t": 0.3}, detail["chains"]
    assert detail["chains"]["B"] == {"t2b": 0.6, "b2t": 0.5}, detail["chains"]
    assert abs(detail["merged_t2b"] - detail["merged_b2t"]) > 0.1, (
        "方向不对称性未体现 — t2b/b2t 不应相等"
    )
    print(f"  multi-chain weighted: t2b={detail['merged_t2b']:.4f} "
          f"b2t={detail['merged_b2t']:.4f} ✓")


def test_merge_single_chain() -> None:
    """单链 target: merged = pair[A][binder] 本身。"""
    m = torch.zeros(2, 2, dtype=torch.float64)
    m[0, 1] = 0.7   # pair[A][binder]
    m[1, 0] = 0.4   # 反方向
    detail = bd.merge_pair_chains_iptm(m, [129])
    assert abs(detail["merged_t2b"] - 0.7) < 1e-9, detail
    assert abs(detail["merged_b2t"] - 0.4) < 1e-9, detail
    print(f"  single-chain: t2b={detail['merged_t2b']:.4f} "
          f"(=pair[A][binder]) ✓")


def test_merge_batch_dim() -> None:
    """[B, n, n] 输入取第一个 batch 元素。"""
    m = torch.zeros(2, 2, 2, dtype=torch.float64)
    m[0, 0, 1] = 0.9
    m[1, 0, 1] = 0.1  # 不应被使用
    detail = bd.merge_pair_chains_iptm(m, [50])
    assert abs(detail["merged_t2b"] - 0.9) < 1e-9, detail
    print("  batch dim: uses element 0 ✓")


def test_remark_roundtrip() -> None:
    """pdb_with_quality_remarks 写入 → _parse_remarks 回读。"""
    annotated = bd.pdb_with_quality_remarks(
        MINIMAL_PDB,
        iptm=0.9,
        pair_chains_iptm=0.6123,
        pair_chains_iptm_b2t=0.5891,
        pair_chains_iptm_chains="A:0.6123,B:0.7",
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as tmp:
        tmp.write(annotated)
        path = tmp.name
    try:
        remarks = _parse_remarks(path)
        assert remarks["iptm"] == 0.9, remarks
        assert abs(remarks["pair_chains_iptm"] - 0.6123) < 1e-9, remarks
        assert abs(remarks["pair_chains_iptm_b2t"] - 0.5891) < 1e-9, remarks
        print("  REMARK roundtrip: pair/b2t 可解析 ✓")
    finally:
        os.unlink(path)


def test_quality_gate() -> None:
    """min_pair_chains_iptm 门: PASS / FAIL / REMARK 缺失三路径。"""
    with_pair = bd.pdb_with_quality_remarks(
        MINIMAL_PDB, pair_chains_iptm=0.55,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as tmp:
        tmp.write(with_pair)
        path_pair = tmp.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as tmp:
        tmp.write(MINIMAL_PDB)
        path_bare = tmp.name
    try:
        # FAIL: 0.55 < 0.70
        cfg = QualityFilterConfig(
            min_iptm=None, min_pair_chains_iptm=0.70, min_binder_plddt=None,
        )
        qc = filter_pdb_by_quality(path_pair, config=cfg)
        assert not qc["passed"], qc
        assert any("pair_chains_iptm=0.5500 < 0.70" in r for r in qc["reasons"]), (
            qc["reasons"]
        )

        # PASS: 0.55 ≥ 0.50
        cfg = QualityFilterConfig(
            min_iptm=None, min_pair_chains_iptm=0.50, min_binder_plddt=None,
        )
        qc = filter_pdb_by_quality(path_pair, config=cfg)
        assert qc["passed"], qc["reasons"]
        assert qc["pair_chains_iptm"] == 0.55, qc

        # REMARK 缺失 → FAIL "not found" (旧行为 PDB 不会被静默放过)
        cfg = QualityFilterConfig(
            min_iptm=None, min_pair_chains_iptm=0.70, min_binder_plddt=None,
        )
        qc = filter_pdb_by_quality(path_bare, config=cfg)
        assert not qc["passed"], qc
        assert any("not found" in r for r in qc["reasons"]), qc["reasons"]
        print("  quality gate: FAIL/PASS/missing 三路径 ✓")
    finally:
        os.unlink(path_pair)
        os.unlink(path_bare)


def test_stage6_metric_switch() -> None:
    """Stage-6 排序/过滤的 iptm_metric 开关 (无 GPU)。"""
    from core import (
        PipelineResult,
        _build_data_dict,
        apply_filters,
        compute_selection_score,
    )

    # 夹具: proxy=0.9 (过 Average_distogram_binding_confidence≥0.8 闸),
    # binder_name 不含 "minibinder" (绕开 pI 闸), 隔离 Average_i_pTM 槽位
    pair_good = PipelineResult(
        binder_sequence="AGSTNDQELM", pair_chains_iptm=0.55, iptm=0.80,
        distogram_iptm_proxy=0.9, target_name="t", binder_name="test_binder",
    )
    pair_bad = PipelineResult(
        binder_sequence="B" * 11, pair_chains_iptm=0.25, iptm=0.88,
        distogram_iptm_proxy=0.9, target_name="t", binder_name="test_binder",
    )
    legacy_only = PipelineResult(
        binder_sequence="C" * 11, iptm=0.70, distogram_iptm_proxy=0.9,
        target_name="t", binder_name="test_binder",
    )  # 旧行无 pair 值 → 回退全局

    # 排序: pair 模式下 pair 值决定同尺度行序 (混入 legacy 回退行的批内
    # minmax 跨尺度不可比 — 生产批次所有行都有 pair 值, 回退仅防异常行)
    ranked = compute_selection_score(
        [pair_bad, pair_good], iptm_metric="pair_chains")
    assert ranked[0] is pair_good, [r.binder_sequence for r in ranked]
    ranked_g = compute_selection_score(
        [pair_bad, pair_good], iptm_metric="global")
    assert ranked_g[0] is pair_bad, [r.binder_sequence for r in ranked_g]
    # legacy 旧行 (无 pair 值): pair 模式回退全局, 不崩溃且得分有限
    ranked_mixed = compute_selection_score(
        [legacy_only], iptm_metric="pair_chains")
    assert 0.0 <= ranked_mixed[0].selection_score <= 1.0

    # 过滤槽位: Average_i_pTM 喂 pair 值; 旧行回退全局
    d = _build_data_dict(pair_good, iptm_metric="pair_chains")
    assert d["Average_i_pTM"] == 0.55, d
    d = _build_data_dict(pair_good, iptm_metric="global")
    assert d["Average_i_pTM"] == 0.80, d
    d = _build_data_dict(legacy_only, iptm_metric="pair_chains")
    assert d["Average_i_pTM"] == 0.70, d

    # apply_filters 透传 (default_filters.json Average_i_pTM 闸)
    passed, _ = apply_filters(
        [pair_good, pair_bad], iptm_metric="pair_chains")
    names = {r.binder_sequence[0] for r in passed}
    assert "A" in names and "B" not in names, (
        f"pair 模式下低 pair 值应被 Average_i_pTM 闸拦截: {names}")
    print("  Stage-6 排序/过滤指标开关 ✓")


def main() -> None:
    print("[test_pair_chains_iptm]")
    test_merge_weighted_t2b()
    test_merge_single_chain()
    test_merge_batch_dim()
    test_remark_roundtrip()
    test_quality_gate()
    test_stage6_metric_switch()
    print("全部通过 ✓")


if __name__ == "__main__":
    main()
