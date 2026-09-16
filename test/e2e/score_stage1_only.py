#!/usr/bin/env python3
"""
stage1-only 单结构离线 Rosetta 评分 — 用于无 selection.parquet 的任务
(如 4ZQK/with: MPNN 后代全灭, 但 stage1 设计本体已过 Stage-2 质量门)。

对 stage1_only/stage1_<hero>.pdb 执行 screen_single_pdb (含 relax),
以 core 原生函数重建 PipelineResult → compute_selection_score →
apply_filters → select_top_n, 重写 design_summary.txt 并注入 REMARK dG。

用法:
  python test/e2e/score_stage1_only.py --task 4ZQK_multi --hotspot with \
      [--seed 1] [--extra-note "..."]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from core import (  # noqa: E402
    PipelineResult,
    apply_filters,
    compute_selection_score,
    select_top_n,
)
from bindercraft_filter.integration import screen_single_pdb  # noqa: E402

TASK_CHAINS = {
    "3RRQ_single": (["A"], "B"),
    "4ZQK_multi": (["A", "B"], "C"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASK_CHAINS))
    ap.add_argument("--hotspot", required=True, choices=["with", "no"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-note", default="")
    args = ap.parse_args()

    target_chains, binder_chain = TASK_CHAINS[args.task]
    base = _pipeline_dir / "test" / "results" / args.task / f"{args.hotspot}_hotspot"
    s1 = pd.read_parquet(base / "stage1_only" / "stage1_results.parquet")
    row = s1.iloc[0]
    pdb_file = base / "stage1_only" / f"stage1_{row['hero_model']}.pdb"
    print(f"[score1] {pdb_file.name} iPTM={row['iptm']:.4f} "
          f"pI={row['isoelectric_point']:.2f}", flush=True)

    t0 = time.time()
    res = screen_single_pdb(
        pdb_path=str(pdb_file),
        binder_sequence=str(row["binder_sequence"]),
        dssp_path=DSSP,
        binder_chain=binder_chain,
        target_chains=list(target_chains),
        relax_first=True,
        omit_AAs="C",
    )
    iface = res["interface_scores"]
    print(f"[score1] relax+评分 {time.time()-t0:.0f}s: "
          f"dG={iface['interface_dG']:.2f} sc={iface['interface_sc']:.2f} "
          f"hbonds={iface['interface_interface_hbonds']} "
          f"unsat={iface['interface_delta_unsat_hbonds']} "
          f"clashes={res['clashes_after_relax']}", flush=True)

    pr_res = PipelineResult(
        target_name=args.task, binder_name="minibinder", stage="esmfold2",
        binder_sequence=str(row["binder_sequence"]),
        iptm=float(row["iptm"]), ptm=float(row["ptm"]) if row["ptm"] else None,
        distogram_iptm_proxy=float(row["distogram_iptm_proxy"]),
        rosetta_dG=iface["interface_dG"], rosetta_sc=iface["interface_sc"],
        rosetta_hbonds=iface["interface_interface_hbonds"],
        rosetta_unsat_hbonds=iface["interface_delta_unsat_hbonds"],
        rosetta_clashes=res["clashes_after_relax"],
        rosetta_binder_energy=iface["binder_score"],
        isoelectric_point=float(row["isoelectric_point"]),
    )
    scored = compute_selection_score([pr_res], rosetta_weight=0.3)
    passed, failed = apply_filters(scored)
    selected = select_top_n(passed, top_n=84)
    r = scored[0]
    is_passed = bool(r.filter_passed)
    print(f"[score1] filter={'PASS' if is_passed else 'FAIL'} "
          f"unmet={list(r.filter_unmet)} selected={len(selected)}", flush=True)

    # REMARK dG 注入 (幂等)
    lines = [ln for ln in pdb_file.read_text().split("\n")
             if "rosetta_dG=" not in ln]
    ins = None
    for i, ln in enumerate(lines):
        if ln.startswith("REMARK 250"):
            ins = i + 1
        elif ln.startswith(("ATOM", "HETATM")):
            break
    lines.insert(ins if ins is not None else 0,
                 f"REMARK 250  rosetta_dG={iface['interface_dG']:.4f}")
    pdb_file.write_text("\n".join(lines))

    def fmt(v, w=8, prec=4):
        return f"{v:>{w}.{prec}f}" if v is not None else " " * (w - 2) + "NA"

    txt = [
        f"# e2e 设计结果 (stage1-only 离线评分; MPNN 后代阴性)  "
        f"task={args.task}  hotspot={args.hotspot}  seed={args.seed}  "
        f"epitope_weight=0.05",
        f"# 生成时间 {time.strftime('%F %T')}  |  Rosetta weight=0.3, "
        f"链 target={target_chains}/binder={binder_chain}",
        f"# 背景: Stage-1 设计过质量门 (iPTM={row['iptm']:.4f}, pLDDT≥0.7, "
        f"pI={row['isoelectric_point']:.2f}); Stage3-6 续跑 100 条 MPNN 后代 "
        f"scaling 3/100 过、hero iPTM 0.55/0.56/0.75、Stage-6 0/3 → 0 selected "
        f"(见 resume_summary.json)。本表 = stage1 设计本体的离线物理评分。",
        f"# 单候选表中 score 为单样本归一化, 不与多候选任务直接可比; "
        f"filter 为阈值判定, 可比。",
        args.extra_note,
        "",
        f"{'rank':>4}  {'stage':<9}  {'iptm':>8}  {'dG':>9}  {'plddt':>8}  "
        f"{'score':>8}  {'filter':<6}  {'sel':>3}  binder_sequence",
        f"{1:>4}  {'esmfold2':<9}  {fmt(float(row['iptm']))}  "
        f"{fmt(iface['interface_dG'], 9, 2)}  {fmt(None)}  "
        f"{fmt(r.selection_score)}  "
        f"{'PASS' if is_passed else 'FAIL':<6}  "
        f"{'Y' if selected else 'N':>3}  {row['binder_sequence']}",
    ]
    (base / "design_summary.txt").write_text("\n".join(txt) + "\n")
    (base / "stage1_only" / "stage1_rosetta_scores.json").write_text(json.dumps({
        "task": args.task, "hotspot": args.hotspot, "seed": args.seed,
        "iptm": float(row["iptm"]),
        "isoelectric_point": float(row["isoelectric_point"]),
        "interface_dG": iface["interface_dG"], "interface_sc": iface["interface_sc"],
        "interface_hbonds": iface["interface_interface_hbonds"],
        "delta_unsat_hbonds": iface["interface_delta_unsat_hbonds"],
        "clashes": res["clashes_after_relax"],
        "binder_score": iface["binder_score"],
        "filter_passed": is_passed, "filter_unmet": list(r.filter_unmet),
        "selected": bool(selected),
    }, indent=2))
    print(f"[score1] 产物: design_summary.txt / "
          f"stage1_only/stage1_rosetta_scores.json / PDB REMARK", flush=True)


if __name__ == "__main__":
    main()
