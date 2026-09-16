#!/usr/bin/env python3
"""冒烟测试: stage1 (critic) 设计进入 Stage-5 Rosetta 队列 + Stage-6 映射。

验证 2026/9/16 的三处 core.py 改动:
  1. _rosetta_score 从 designed_sequence 回退推导链名/序列 (critic 条目)
  2. run() Stage 5 并入 critic_results (此处直接调用该方法验证行为)
  3. _collect_results (A) 分支映射 rosetta_* 字段进 PipelineResult

CPU-only (Rosetta relax ~5-7 min); 复用 4ZQK/with seed1 的 stage1 产物,
预期 dG ≈ −47.67 (与 score_stage1_only.py 同结构同流程)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import pyrosetta as pr  # noqa: E402

DSSP = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"
DALPHABALL = "/home/taoyr/projects/BindCraft-main/functions/DAlphaBall.gcc"
pr.init(f"-holes:dalphaball {DALPHABALL} -mute all")

import pandas as pd  # noqa: E402
from Bio import SeqIO  # noqa: E402
import re  # noqa: E402

from core import (  # noqa: E402
    IntegratedPipeline,
    PipelineConfig,
    _collect_results,
    apply_filters,
    compute_selection_score,
)

DATA = _pipeline_dir / "test" / "data"
base = _pipeline_dir / "test" / "results" / "4ZQK_multi" / "with_hotspot"

seqs = {}
for r in SeqIO.parse(DATA / "rcsb_pdb_4ZQK.fasta", "fasta"):
    m = re.search(r"\|Chain (\S+)\|", r.description)
    seqs[m.group(1)] = str(r.seq).replace("X", "")
target_seq = f"{seqs['A']}|{seqs['B']}"

row = pd.read_parquet(base / "stage1_only" / "stage1_results.parquet").iloc[0]
pdb_file = base / "stage1_only" / f"stage1_{row['hero_model']}.pdb"
critic_entry = {
    # 刻意不带 "target_seq"/"binder_sequence"/"sequence" — 走回退推导
    "designed_sequence": f"{target_seq}|{row['binder_sequence']}",
    "critic_name": str(row["hero_model"]),
    "iptm": float(row["iptm"]),
    "ptm": float(row["ptm"]) if row["ptm"] else None,
    "distogram_iptm_proxy": float(row["distogram_iptm_proxy"]),
    "annotated_pdb": pdb_file.read_text(),
    "complex": object(),   # (A) 分支要求非 None; 冒烟测试用占位对象
    "is_scaling_critic": False,
}

cfg = PipelineConfig(
    enable_rosetta=True,
    rosetta_dssp_path=DSSP,
    rosetta_relax_first=True,
    verbose=True,
)
pipeline = IntegratedPipeline(cfg)

print("[smoke] Stage-5: _rosetta_score([critic_entry]) ...", flush=True)
scored = pipeline._rosetta_score([critic_entry])
screening = scored[0].get("rosetta_screening")
assert screening is not None, "rosetta_screening 未产出 — 回退推导失败!"
iface = screening["interface_scores"]
print(f"[smoke] dG={iface['interface_dG']:.2f} sc={iface['interface_sc']:.2f} "
      f"hbonds={iface['interface_interface_hbonds']} "
      f"unsat={iface['interface_delta_unsat_hbonds']}", flush=True)

results = _collect_results(
    critic_results=[critic_entry], quality_passed=[], refold_results=[],
    target_name="4ZQK_multi", binder_name="minibinder", seed=1,
    is_antibody=False,
)
pr = results[0]
print(f"[smoke] PipelineResult: stage={pr.stage} rosetta_dG={pr.rosetta_dG} "
      f"rosetta_sc={pr.rosetta_sc} hbonds={pr.rosetta_hbonds}", flush=True)
assert pr.rosetta_dG is not None, "(A) 分支未映射 rosetta_dG!"

scored_prs = compute_selection_score(results, rosetta_weight=0.3)
passed, failed = apply_filters(scored_prs)
print(f"[smoke] Stage-6: score={scored_prs[0].selection_score:.4f} "
      f"filter_passed={scored_prs[0].filter_passed} "
      f"unmet={list(scored_prs[0].filter_unmet)}", flush=True)
print("[smoke] 全部断言通过 ✅")
