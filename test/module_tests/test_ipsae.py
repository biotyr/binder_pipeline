#!/usr/bin/env python3
"""ipsae.py 自实现模块的测试 (CPU, 无模型)。

[等价性] test_ipsae_equivalence: 对参考实现 (IPSAE-main/ipsae.py) 的
  Example 数据 (AF2 multimer 三链复合体, cutoff 15/15) 做逐链对逐方向
  数值等价断言 (绝对误差 < 1e-6, 主导项是参考脚本 txt 的 6 位打印舍入)。
  这是自实现路线 (B) 的正确性基石 —— 修改 ipsae.py 或参考脚本必须重跑。

[单测] 方向不对称 (t2b≠b2t)、cutoff 语义 (全低 PAE→按公式可解析值 /
  全高 PAE→0)、batch 维、compute_ipsae 链 mask 边界 (锚必须在 target 内、
  binder 段尾部)。

用法: python test/module_tests/test_ipsae.py
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from ipsae import (  # noqa: E402
    IPSAE_PAE_CUTOFF,
    _d0_from_counts,
    _ptm,
    compute_ipsae,
    ipsae_d0res_pair,
)

IPSAE_REF = Path("/home/taoyr/projects/IPSAE-main/ipsae.py")
EXAMPLE_DIR = Path("/home/taoyr/projects/IPSAE-main/Example")
EXAMPLE_JSON_GZ = EXAMPLE_DIR / (
    "RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json.gz"
)
EXAMPLE_PDB = EXAMPLE_DIR / (
    "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000.pdb"
)


# ---- 等价性测试 ----

def _parse_reference_txt(txt_path: Path) -> dict[tuple[str, str], float]:
    """解析 ipsae.py 输出 txt: {(Chn1, Chn2): ipSAE} (仅 asym 行)。"""
    out: dict[tuple[str, str], float] = {}
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[4] != "asym":
            continue
        out[(parts[0], parts[1])] = float(parts[5])
    return out


def _parse_pdb_chains(pdb_path: Path) -> list[str]:
    """PDB CA 行的链归属 (文件序 = AF2 token 序)。"""
    chains: list[str] = []
    for ln in pdb_path.read_text().splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            chains.append(ln[21])
    return chains


def test_ipsae_equivalence() -> None:
    assert IPSAE_REF.exists(), f"参考实现不存在: {IPSAE_REF}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        json_path = tmp / "scores.json"
        with gzip.open(EXAMPLE_JSON_GZ, "rb") as fin, \
                open(json_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        pdb_path = tmp / EXAMPLE_PDB.name
        shutil.copy(EXAMPLE_PDB, pdb_path)

        subprocess.run(
            [sys.executable, str(IPSAE_REF), str(json_path),
             str(pdb_path), "15", "15"],
            check=True, capture_output=True,
        )
        ref = _parse_reference_txt(tmp / (pdb_path.stem + "_15_15.txt"))
        assert ref, "参考输出 txt 未解析到 asym 行"

        pae = np.asarray(json.load(open(json_path))["pae"], dtype=float)
        chains = _parse_pdb_chains(pdb_path)
        assert pae.shape == (len(chains), len(chains)), (
            f"PAE {pae.shape} 与 PDB 残基数 {len(chains)} 不一致"
        )
        idx = np.arange(len(chains))

        n_checked = 0
        for (c1, c2), ref_val in sorted(ref.items()):
            got, _, _, _ = ipsae_d0res_pair(
                pae, chains == np.asarray(c1), chains == np.asarray(c2),
                cutoff=15.0,
            )
            assert abs(got - ref_val) < 1e-6, (
                f"链对 {c1}->{c2}: 自实现 {got:.8f} vs 参考 {ref_val:.8f}"
            )
            n_checked += 1
        assert n_checked >= 6, f"等价链对数异常少: {n_checked}"
        del idx
        print(f"  等价性: {n_checked} 个链对 × cutoff15 全部 <1e-6 ✓")


# ---- 单测 ----

def test_target_internal_invariance() -> None:
    """target 内部链对不进入计算: 改 target-target 块 → t2b/b2t 逐位不变。

    钉死 "避开 target 对 binder 质量评估的干扰" 性质 (与 pair_chains_iptm
    同语义): 两套指标的算术域均为 binder↔target 对, target 内部 PAE 既不进
    n0res 也不进均值 (列掩码严格限定 binder)。
    """
    rng = np.random.default_rng(0)
    pae_a = rng.uniform(0.5, 30.0, size=(130, 130))
    pae_a[:50, 100:] = 3.0
    pae_a[100:, :100] = 10.0
    # 独立重采样 target-target 块 (行/列均 <100)
    pae_b = pae_a.copy()
    pae_b[:100, :100] = rng.uniform(0.5, 30.0, size=(100, 100))

    d_a = compute_ipsae(pae_a, [100])
    d_b = compute_ipsae(pae_b, [100])
    assert d_a["ipsae_t2b"] == d_b["ipsae_t2b"], (
        f"target 内部块变动改变了 t2b: {d_a['ipsae_t2b']} vs {d_b['ipsae_t2b']}"
    )
    assert d_a["ipsae_b2t"] == d_b["ipsae_b2t"]
    assert d_a["anchor_t2b"] == d_b["anchor_t2b"]
    for k in d_a["chains"]:
        assert d_a["chains"][k] == d_b["chains"][k]
    print("  target 内部链对不变性: t2b/b2t/锚逐位不变 ✓")


def test_direction_and_cutoff() -> None:
    pae = np.full((130, 130), 25.0)
    pae[:50, 100:] = 3.0
    pae[50:100, 100:] = 22.0
    pae[100:, :100] = 10.0

    idx = np.arange(130)
    tgt, bnd = idx < 100, idx >= 100
    t2b, anchor, n0, _ = ipsae_d0res_pair(pae, tgt, bnd)
    b2t, _, _, _ = ipsae_d0res_pair(pae, bnd, tgt)

    # t2b: 最好锚在前 50 行 (n0res=30), 锚必须落在 target 段
    assert 0 <= anchor < 50, anchor
    assert n0 == 30, n0
    exp_d0 = _d0_from_counts(np.array([30.0]))[0]
    exp_score = float(_ptm(np.array([3.0]), exp_d0)[0])
    assert abs(t2b - exp_score) < 1e-12, (t2b, exp_score)
    # 方向不对称: b2t 锚在 binder 行, PAE=10 → 分数不同
    assert abs(t2b - b2t) > 1e-3, (t2b, b2t)
    print(f"  方向/锚段: t2b={t2b:.4f} (anchor@{anchor}, n0res={n0}) "
          f"b2t={b2t:.4f} ✓")

    # 全高 PAE → 0 分
    hi, _, n0_hi, _ = ipsae_d0res_pair(
        np.full((130, 130), 25.0), tgt, bnd)
    assert hi == 0.0 and n0_hi == 0, (hi, n0_hi)
    print("  cutoff 语义: 全高 PAE → 0 ✓")


def test_compute_ipsae_wrapper() -> None:
    pae = np.full((130, 130), 25.0)
    pae[:50, 100:] = 3.0
    detail = compute_ipsae(pae, [100], cutoff=IPSAE_PAE_CUTOFF)
    assert detail["anchor_t2b"] < 100, "锚必须落在 target 段"
    assert detail["chains"]["A"]["t2b"] == detail["ipsae_t2b"], (
        "单链 target 时逐链值应与合并值一致"
    )
    # batch 维: 取第一个元素
    detail_b = compute_ipsae(pae[None], [100])
    assert detail_b["ipsae_t2b"] == detail["ipsae_t2b"]
    # torch tensor 输入 (无 import torch, 用 duck-typing 路径)
    class _FakeTensor:
        def __init__(self, a): self._a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._a
    detail_t = compute_ipsae(_FakeTensor(pae), [100])
    assert detail_t["ipsae_t2b"] == detail["ipsae_t2b"]
    print(f"  wrapper: t2b={detail['ipsae_t2b']:.4f} "
          f"anchor={detail['anchor_t2b']} batch/tensor ✓")


def main() -> None:
    print("[test_ipsae]")
    test_target_internal_invariance()
    test_direction_and_cutoff()
    test_compute_ipsae_wrapper()
    if EXAMPLE_JSON_GZ.exists() and EXAMPLE_PDB.exists():
        test_ipsae_equivalence()
    else:
        print("  [skip] IPSAE-main Example 数据不存在, 等价性测试跳过")
    print("全部通过 ✓")


if __name__ == "__main__":
    main()
