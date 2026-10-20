#!/usr/bin/env python3
"""
pair_chains_iptm 回放定标: 对已判定的真解/退化产物做 hero 200 步重折,
提取 binder-target pair_chains_iptm (t2b/b2t/逐链) + 全局 iptm 对照,
为 Stage-2/6 阈值切换 (global → pair_chains) 提供定标数据。

背景 (2026-10-19 决策, 见 memory pipeline-pair-iptm-metric-decision):
置信度指标源从全局 ipTM 切换到 binder-target pair_chains_iptm
([target][binder] t2b 方向, 残基加权合并 = binder×target 子块均值)。
现阈值 0.50 (单链) / 0.70 (多链) / Average_i_pTM 0.5 是对全局
max-over-rows 值标定的, 块均值系统性偏低 → 翻转默认前必须重标定。

产物清单 (全部为已归档的真实产物, 序列从 parquet/result.json 读取):
  founder    4ZQK junction stage1 (iPTM 0.8245, dG -47.67, 真锚定)  真解基线/多链
  nohot_4z   4ZQK no_hotspot stage1 (iPTM 0.9077)                   真解基线/多链
  nohot_3r   3RRQ no_hotspot stage1 (iPTM 0.8597)                   真解基线/单链
  groov_3r   3RRQ with_hotspot_groove stage1 (iPTM 0.8575)          真解基线/单链
  seed2/3/4  3g junction 多 seed (iPTM 0.596/0.779/0.685)           质量谱系/多链
             (seed3 曾以 0.779 全线过质量门 — pair 值应正常, 验证
              "传递置信度合法" 预期; 热点 honoring 由几何门另行拦截)
  wrapper    diag_len158/w0.5 重采样 (iPTM 0.9003, S/R 低复杂度)    退化对照/多链

注意: 重折 seed=0 统一 (与原评分轮的 seed 不同, 结构有轻微采样差,
定标的是指标语义而非逐位复现)。

用法: CUDA_VISIBLE_DEVICES=<GPU-UUID> python diag_pair_iptm_replay.py
      REPLAY_LABELS=founder,wrapper ... (子集重跑)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
for p in (str(_pipeline_dir), str(_pipeline_dir / "test" / "e2e")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

import run_e2e  # noqa: E402,F401  (Patch A + pyrosetta.init + hero 限 1)
import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import merge_pair_chains_iptm  # noqa: E402

OUT_DIR = _pipeline_dir / "test" / "module_results" / "diag_pair_iptm_replay"
HERO = "ESMFold2-Experimental-Cutoff2025"
RESULTS = _pipeline_dir / "test" / "results"
MODULE_RESULTS = _pipeline_dir / "test" / "module_results"


def _stage1_binder(task_dir: str) -> dict:
    """从 e2e stage1 parquet 读第一行 hero 序列与原始指标。"""
    df = pd.read_parquet(RESULTS / task_dir / "stage1_only" / "stage1_results.parquet")
    row = df.iloc[0]
    return {
        "binder_seq": row["binder_sequence"],
        "orig_iptm": float(row["iptm"]),
        "src": f"{task_dir}/stage1_only (parquet row 0)",
    }


def _3g_seed(seed: int) -> dict:
    d = json.load(open(MODULE_RESULTS / "diag_inter_schedule3g_junction"
                       / f"seed{seed}" / "result.json"))
    return {
        "binder_seq": d["binder_sequence"],
        "orig_iptm": float(d["iptm"]),
        "src": f"diag_inter_schedule3g_junction/seed{seed}",
    }


def build_manifest() -> dict[str, dict]:
    m: dict[str, dict] = {}
    m["founder"] = {**_stage1_binder("4ZQK_multi/with_hotspot_junction"),
                    "task": "4ZQK_multi", "class": "good_multi",
                    "note": "junction founder, dG-47.67 真锚定"}
    m["nohot_4z"] = {**_stage1_binder("4ZQK_multi/no_hotspot"),
                     "task": "4ZQK_multi", "class": "good_multi",
                     "note": "4ZQK 无热点 (iPTM 0.9077)"}
    m["nohot_3r"] = {**_stage1_binder("3RRQ_single/no_hotspot"),
                     "task": "3RRQ_single", "class": "good_single",
                     "note": "3RRQ 无热点 (iPTM 0.8597)"}
    m["groov_3r"] = {**_stage1_binder("3RRQ_single/with_hotspot_groove"),
                     "task": "3RRQ_single", "class": "good_single",
                     "note": "3RRQ 凹槽热点 (iPTM 0.8575)"}
    for seed in (2, 3, 4):
        m[f"seed{seed}"] = {**_3g_seed(seed), "task": "4ZQK_multi",
                            "class": "quality_ok_multi",
                            "note": f"3g junction seed{seed} (质量合格, "
                                    f"seed3 热点假阳性)"}
    w = json.load(open(MODULE_RESULTS / "diag_len158" / "w0.5" / "result.json"))
    m["wrapper"] = {"binder_seq": w["designed_sequence"].split("|")[-1],
                    "orig_iptm": float(w["iptm"]),
                    "src": "diag_len158/w0.5/result.json",
                    "task": "4ZQK_multi",
                    "class": "degraded_multi",
                    "note": "S/R 低复杂度包裹型 (iPTM 0.9003, pLDDT 门拦截)"}
    return m


def main() -> None:
    manifest = build_manifest()
    only = os.environ.get("REPLAY_LABELS")
    if only:
        keep = {s.strip() for s in only.split(",") if s.strip()}
        manifest = {k: v for k, v in manifest.items() if k in keep}

    targets = {task: run_e2e.load_target(task) for task in
               {v["task"] for v in manifest.values()}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "result.json"
    rows: dict[str, dict] = {}
    if out_path.exists():  # 断点续跑
        rows = json.load(open(out_path))

    print(f"[replay] {len(manifest)} 条待重折: {sorted(manifest)}")
    model = bd._load_hf_model(HERO, lm_dropout=0.25, cache_esmc=True,
                              device="cuda")
    try:
        for label, item in manifest.items():
            if label in rows:
                print(f"[replay] {label}: 已有结果, 跳过")
                continue
            target_seq, _ = targets[item["task"]]
            chain_lens = [len(c) for c in target_seq.split("|")]
            binder_seq = item["binder_seq"]
            t0 = time.time()
            fold = bd.fold_and_get_distogram(
                model,
                target_seq,
                bd.build_target_one_hot(target_seq),
                bd.sequence_to_one_hot(binder_seq)[..., 2:22],
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=0,
            )
            iptm_t = fold.get("iptm")
            iptm = float(iptm_t.item()) if iptm_t is not None else None
            pair_matrix = fold["output"].get("pair_chains_iptm")
            detail = merge_pair_chains_iptm(pair_matrix, chain_lens) \
                if pair_matrix is not None else None
            rows[label] = {
                "class": item["class"], "note": item["note"],
                "src": item["src"], "orig_iptm": item["orig_iptm"],
                "binder_len": len(binder_seq),
                "chain_lens": chain_lens,
                "global_iptm_replay": iptm,
                "merged_t2b": detail["merged_t2b"] if detail else None,
                "merged_b2t": detail["merged_b2t"] if detail else None,
                "chains": detail["chains"] if detail else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
            json.dump(rows, open(out_path, "w"), indent=2, ensure_ascii=False)
            print(f"[replay] {label}: global={iptm:.4f} "
                  f"t2b={detail['merged_t2b']:.4f} "
                  f"b2t={detail['merged_b2t']:.4f} "
                  f"({item['class']}, {rows[label]['elapsed_s']}s)")
            del fold
    finally:
        del model
        import torch
        torch.cuda.empty_cache()

    # ---- 汇总表 + 阈值建议 ----
    print("\n[replay] 汇总 (按 replay t2b 降序):")
    header = (f"{'label':<10} {'class':<17} {'t2b':>7} {'b2t':>7} "
              f"{'global':>7} {'orig':>7}")
    print(header)
    ordered = sorted(rows.items(),
                     key=lambda kv: kv[1]["merged_t2b"] or 0.0, reverse=True)
    for label, r in ordered:
        print(f"{label:<10} {r['class']:<17} "
              f"{r['merged_t2b']:>7.4f} {r['merged_b2t']:>7.4f} "
              f"{r['global_iptm_replay']:>7.4f} {r['orig_iptm']:>7.4f}")

    def _min_in(class_key: str) -> float | None:
        vals = [r["merged_t2b"] for r in rows.values()
                if r["class"] == class_key and r["merged_t2b"] is not None]
        return min(vals) if vals else None

    good_multi = _min_in("good_multi")
    good_single = _min_in("good_single")
    degraded = _min_in("degraded_multi")
    print("\n[replay] 阈值建议 (真解全过 + 退化拦截):")
    if good_multi is not None:
        print(f"  多链 quality_min_iptm_multi → ≤{good_multi:.4f} "
              f"(good_multi 最小 t2b)"
              + (f"; 退化样本 t2b={degraded:.4f}"
                 f" → 取区间 ({degraded:.2f}, {good_multi:.2f}) 内" if degraded is not None else ""))
    if good_single is not None:
        print(f"  单链 quality_min_iptm_single → ≤{good_single:.4f} "
              f"(good_single 最小 t2b)")
    print(f"  Average_i_pTM 槽 (default_filters.json) 与多链门同理下移; "
          f"最终取值由用户拍板")
    print(f"\n[replay] 结果已写 {out_path}")


if __name__ == "__main__":
    main()
