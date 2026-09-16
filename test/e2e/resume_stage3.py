#!/usr/bin/env python3
"""
Stage-3 续跑驱动 — 跳过 Stage 1/2, 基于已落盘的 stage1 结果直接投入
MPNN(扩量) → scaling(阈值不变) → hero 重折叠 → Rosetta → Stage 6。

背景 (2026-09-16): seed1 热点设计首次过 Stage-2 质量门 (iPTM 0.8245),
但其 20 条 MPNN 序列在 scaling 快筛 0/20 通过 (proxy 0.32-0.45 < 0.50)。
用户方案: 不放松过滤, 扩大 MPNN 采样量再筛。

复用 run_e2e 的环境补丁 (import 即生效: Patch A + pyrosetta init)。
说明: 绕过 _mpnn_redesign_passed 是因其上游 mpnn_redesign_from_critic_results
丢弃 complex=None 的条目 (resume 场景无内存 complex 对象), 直接调
mpnn_redesign_binder (生产同一落点函数, 显式传链名 C/[A,B])。
stage1 设计本体不进本 run 的 Stage 6 (与生产一致由 rescore 统一补入)。

用法:
  CUDA_VISIBLE_DEVICES=<UUID> python test/e2e/resume_stage3.py \
      --task 4ZQK_multi --hotspot with --seed 1 --mpnn-num-seqs 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import torch  # noqa: E402

# run_e2e 导入即完成: 环境变量 / Patch A / pyrosetta.init
import run_e2e  # noqa: E402, F401
from run_e2e import DSSP, load_target  # noqa: E402

from esmcp.binder_design import _apply_torch_compile, _load_hf_model  # noqa: E402
from inverse_optimize.mpnn_redesign import mpnn_redesign_binder  # noqa: E402

import pandas as pd  # noqa: E402

from core import (  # noqa: E402
    IntegratedPipeline,
    PipelineConfig,
    _collect_results,
    _save_outputs,
    apply_filters,
    compute_selection_score,
    select_top_n,
)

RESULTS_ROOT = _pipeline_dir / "test" / "results"
HERO = "ESMFold2-Experimental-Cutoff2025"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="4ZQK_multi")
    ap.add_argument("--hotspot", default="with", choices=["with", "no"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mpnn-num-seqs", type=int, default=100)
    args = ap.parse_args()

    task_dir = RESULTS_ROOT / args.task / f"{args.hotspot}_hotspot"
    full_dir = task_dir / "full_pipeline"
    s1_dir = task_dir / "stage1_only"

    s1 = pd.read_parquet(s1_dir / "stage1_results.parquet")
    row = s1.iloc[0]
    pdb_file = s1_dir / f"stage1_{row['hero_model']}.pdb"
    assert pdb_file.exists(), pdb_file
    target_seq, chain_order = load_target(args.task)
    target_seqs = target_seq.split("|")
    # 链名标签 (asym_id 顺序): target 链 A,B..., binder 紧随其后 — 与
    # _derive_chain_metadata/rescore 同约定; mpnn_redesign_binder 按
    # structure[0][label] 索引, 传标签而非序列
    target_labels = [chr(ord("A") + i) for i in range(len(target_seqs))]
    binder_label = chr(ord("A") + len(target_seqs))
    print(f"[resume] stage1: iPTM={row['iptm']:.4f} "
          f"proxy={row['distogram_iptm_proxy']:.4f} "
          f"pI={row['isoelectric_point']:.2f} pdb={pdb_file.name}")
    print(f"[resume] target_len={len(target_seq.replace('|',''))} "
          f"chains={target_labels}/{binder_label} mpnn_num_seqs={args.mpnn_num_seqs}")

    cfg = PipelineConfig(
        enable_mpnn=True,
        enable_rosetta=True,
        rosetta_dssp_path=DSSP,
        rosetta_relax_first=True,
        mpnn_num_seqs=args.mpnn_num_seqs,
        hero_critic_names=[HERO],
        top_n=84,
        output_dir=str(full_dir),
        save_pdbs=True,
        verbose=True,
        scaling_threshold=0.50,   # 用户要求: 不放松过滤
    )
    pipeline = IntegratedPipeline(cfg)

    t0 = time.time()
    # ── Stage 3: MPNN 逆折叠 (直接调生产落点函数, 100 条采样) ──
    designs = mpnn_redesign_binder(
        pdb_path=str(pdb_file),
        binder_chain=binder_label,
        target_chains=target_labels,
        num_seqs=args.mpnn_num_seqs,
        sampling_temp=cfg.mpnn_sampling_temp,
        fix_interface=cfg.mpnn_fix_interface,
    )
    print(f"[resume] Stage 3: {len(designs)} MPNN 序列 "
          f"({time.time()-t0:.0f}s)")
    mpnn_results = [{
        "original_designed_sequence": f"{target_seq}|{row['binder_sequence']}",
        "critic_name": row["hero_model"],
        "iptm": row["iptm"], "ptm": row["ptm"],
        "mpnn_designs": designs,
    }]

    # ── Stage 4: scaling(0.50) + hero 重折叠 ──
    hero = _load_hf_model(HERO, lm_dropout=0.25, cache_esmc=True, device="cuda")
    _apply_torch_compile(hero)
    refold_results = pipeline._refold_mpnn_designs(
        mpnn_results, target_seq, False, hero_models={HERO: hero})
    del hero
    torch.cuda.empty_cache()

    # ── Stage 5: Rosetta 物理评分 ──
    if cfg.enable_rosetta and refold_results:
        refold_results = pipeline._rosetta_score(refold_results)

    # ── Stage 6: 打分/过滤/选择/落盘 (stage1 本体由 rescore 统一补入) ──
    all_results = _collect_results(
        critic_results=[], quality_passed=[], refold_results=refold_results,
        target_name=args.task, binder_name="minibinder",
        seed=args.seed, is_antibody=False,
    )
    all_results = compute_selection_score(all_results, rosetta_weight=0.3)
    passed, failed = apply_filters(all_results)
    selected = select_top_n(passed, top_n=cfg.top_n)
    if cfg.verbose:
        print(f"[resume] Filter: {len(passed)} passed / {len(failed)} failed")
        print(f"[resume] Selected: Top-{cfg.top_n} = {len(selected)} designs")
    if selected and cfg.save_pdbs:
        _save_outputs(selected, cfg)

    n_scaling_pass = len(refold_results)
    summary = {
        "task": args.task, "hotspot": args.hotspot, "seed": args.seed,
        "mpnn_num_seqs": args.mpnn_num_seqs, "scaling_threshold": 0.50,
        "n_mpnn": len(designs), "n_scaling_passed": n_scaling_pass,
        "n_filter_passed": len(passed), "n_selected": len(selected),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Stage3-6 续跑: 复用 stage1_only 的已过质量门设计 (seed1, "
                "iPTM 0.8245), scaling 阈值维持 0.50 未放松",
    }
    (task_dir / "resume_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[resume] DONE — {task_dir / 'resume_summary.json'}")


if __name__ == "__main__":
    main()
