#!/usr/bin/env python3
"""
pair_chains_iptm 全链路冒烟 (30 步缩短设计, 单卡 ~5min):

  design_binder → score_critic 提取 pair 值 → REMARK 250 写入 →
  Stage-2 质量门 (iptm_metric="pair_chains") → Stage-6 排序/过滤

检查点:
  1. critic_results (hero 条目) 携带 pair_chains_iptm / ipsae / detail,
     REMARK 行在 (pair + ipsae);
  2. Stage-2 打印行显示 pair_chains_iptm 槽位;
  3. PipelineResult / selection.parquet 携带 pair_chains_iptm 与 ipsae 列;
  4. 全链路无异常 (设计质量本身不作断言 — 30 步仅冒烟)。

用法: CUDA_VISIBLE_DEVICES=<GPU-UUID> python smoke_pair_iptm.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
for p in (str(_pipeline_dir), str(_pipeline_dir / "test" / "e2e")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_e2e  # noqa: E402,F401  (Patch A + pyrosetta.init + hero 限 1)
import esmcp.binder_design as bd  # noqa: E402

bd.STEPS = 30  # 冒烟: 缩短日程 (只验证链路, 不验证设计质量)

from core import IntegratedPipeline, PipelineConfig  # noqa: E402

OUT_DIR = _pipeline_dir / "test" / "module_results" / "smoke_pair_iptm"


def main() -> None:
    target_seq, _ = run_e2e.load_target("3RRQ_single")
    cfg = PipelineConfig(
        enable_mpnn=False,
        enable_rosetta=False,
        hero_critic_names=["ESMFold2-Experimental-Cutoff2025"],
        iptm_metric="pair_chains",
        output_dir=str(OUT_DIR),
        save_pdbs=True,
        verbose=True,
    )
    pipe = IntegratedPipeline(cfg)
    result = pipe.run(
        target_name="smoke_3rrq",           # 非 preset 名 → 走显式序列
        binder_name="minibinder",
        target_sequence=target_seq,
        seed=0,
        batch_size=1,
    )

    print("\n[smoke] ---- 检查点 ----")
    heroes = [r for r in result["critic_results"]
              if not r.get("is_scaling_critic")]
    assert heroes, "无 hero critic 条目"
    for r in heroes:
        pair = r.get("pair_chains_iptm")
        assert pair is not None, f"critic_results 未携带 pair 值: {r.keys()}"
        assert r.get("pair_chains_iptm_detail") is not None
        ipsae = r.get("ipsae")
        assert ipsae is not None, f"critic_results 未携带 ipsae: {r.keys()}"
        assert r.get("ipsae_detail") is not None
        pdb = r.get("annotated_pdb") or ""
        assert "pair_chains_iptm=" in pdb, "REMARK 250 缺 pair 行"
        assert "pair_chains_iptm_chains=" in pdb, "REMARK 250 缺逐链行"
        assert "ipsae=" in pdb, "REMARK 250 缺 ipsae 行"
        assert "ipsae_anchor=" in pdb, "REMARK 250 缺 ipsae_anchor 行"
        print(f"[smoke] 1. critic {r['critic_name']}: iptm={r.get('iptm'):.4f} "
              f"pair={pair:.4f} ipsae={ipsae:.4f}  REMARK ✓")

    # Stage-2 打印行 (verbose 输出已在上方日志) + 条目结构
    for r in result["quality_passed"] + result["quality_failed"]:
        qc = r.get("quality_check", {})
        assert "pair_chains_iptm" in qc, f"quality_check 缺 pair 键: {qc.keys()}"
        assert "ipsae" in qc, f"quality_check 缺 ipsae 键: {qc.keys()}"
    print(f"[smoke] 2. quality_check 携带 pair/ipsae 键 ✓ "
          f"(passed={len(result['quality_passed'])} "
          f"failed={len(result['quality_failed'])})")

    fr = result["final_results"]
    assert fr and all(r.pair_chains_iptm is not None for r in fr), (
        "PipelineResult 缺 pair_chains_iptm"
    )
    assert fr and all(r.ipsae is not None for r in fr), (
        "PipelineResult 缺 ipsae"
    )
    sel_parquet = OUT_DIR / "selection.parquet"
    if sel_parquet.exists():
        import pandas as pd
        cols = pd.read_parquet(sel_parquet).columns
        assert "pair_chains_iptm" in cols, f"parquet 缺 pair 列: {list(cols)}"
        assert "ipsae" in cols, f"parquet 缺 ipsae 列: {list(cols)}"
        print("[smoke] 3. PipelineResult + selection.parquet pair/ipsae 列 ✓")
    print(f"[smoke] 4. 全链路冒烟通过 ✓  (全局/逐结果数值见上方日志)")


if __name__ == "__main__":
    main()
