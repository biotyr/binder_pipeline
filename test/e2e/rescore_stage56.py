#!/usr/bin/env python3
"""
Stage 5/6 离线重评分 (补救后备) — 基于已有重折叠结构, 不重跑 Stage 1-4。

定位: run_e2e.py 本身已是完整 pipeline (含修复后的原生 Stage 5/6)。
本脚本仅当 pipeline 内 Stage 5 因链名等问题未产出 dG 时作为补救,
与任务一当时的情况对应 (bug: 链序列被当作链名传入 screen_single_pdb)。

对 run 产物中的全部候选 (stage1 原始设计 + MPNN 重折叠 selected)
执行 Rosetta 物理评分 (screen_single_pdb, 含 FastRelax), 然后按
core.py 的原生 Stage 6 逻辑重建打分与排序:
  compute_selection_score(rosetta_weight=0.3) → apply_filters → select_top_n

覆盖原 design_summary.txt / selection 结果, 并向 PDB 注入 REMARK dG
(幂等: 注入前剔除旧 rosetta_dG 行, 防止与 run_e2e.py 的注入重复)。

用法:
  python test/e2e/rescore_stage56.py --task 3RRQ_single --hotspot no
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

DSSP = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"
DALPHABALL = "/home/taoyr/projects/BindCraft-main/functions/DAlphaBall.gcc"
N_WORKERS = 6
ROSETTA_WEIGHT = 0.3

# 链名约定: target 链按 asym_id 顺序 A,B,...; binder 恒为末链
# (core._derive_chain_metadata)。与重折叠 PDB 的实际链名一致。
TASK_CHAINS = {
    "3RRQ_single": (["A"], "B"),
    "4ZQK_multi": (["A", "B"], "C"),
}

# ---------------- worker ----------------

def _worker_init():
    global _pr
    import pyrosetta as pr
    pr.init(f"-holes:dalphaball {DALPHABALL} -mute all")
    _pr = pr

def _score_one(pdb_path: str, binder_sequence: str, stage: str,
               binder_chain: str, target_chains: list[str]) -> dict:
    from bindercraft_filter.integration import screen_single_pdb
    res = screen_single_pdb(
        pdb_path=pdb_path,
        binder_sequence=binder_sequence,
        dssp_path=DSSP,
        binder_chain=binder_chain,
        target_chains=list(target_chains),
        relax_first=True,
        omit_AAs="C",
    )
    iface = res["interface_scores"]
    return {
        "stage": stage, "pdb": pdb_path,
        "dG": iface["interface_dG"], "sc": iface["interface_sc"],
        "hbonds": iface["interface_interface_hbonds"],
        "unsat": iface["interface_delta_unsat_hbonds"],
        "clashes": res["clashes_after_relax"],
        "binder_energy": iface["binder_score"],
    }

# ---------------- main ----------------

def main() -> None:
    import pandas as pd
    from core import PipelineResult, _compute_pI
    from core import apply_filters, compute_selection_score, select_top_n

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASK_CHAINS), required=True)
    ap.add_argument("--hotspot", choices=["with", "no"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    target_chains, binder_chain = TASK_CHAINS[args.task]
    base = _pipeline_dir / "test" / "results" / args.task / f"{args.hotspot}_hotspot"

    sel = pd.read_parquet(base / "full_pipeline" / "selection.parquet")
    s1 = pd.read_parquet(base / "stage1_only" / "stage1_results.parquet")
    sel_dir = base / "full_pipeline" / "selected_structures"

    # ---- 候选清单: N mpnn + 1 stage1 ----
    jobs = []   # (key, pdb_path, binder_sequence, stage, iptm, proxy, pI)
    for _, r in sel.iterrows():
        rank = int(r["rank"])
        ##### claude 2026/9/15 fix #####
        # selection.parquet 可能含 esmfold2 行 (stage1 设计本体: 其 Rosetta
        # 指标为空使 Rosetta 判据在管线内被跳过, 从而可入选 selected)。
        # 旧代码把每行都硬标 "mpnn" 再额外加 stage1#1 → 同一结构被 relax
        # 评分两次, 统一表出现两行同序列。esmfold2 行一律跳过, 由下方
        # stage1#1 作业统一承担。
        stage_row = str(r["stage"]) if "stage" in sel.columns else "mpnn"
        if stage_row != "mpnn":
            continue
        ##### claude 2026/9/15 fix end #####
        # sorted(): 同 rank 存在普通 PDB 与 *_relaxed.pdb 时, '.'<'_' 保证取普通版
        cands = sorted(sel_dir.glob(f"{rank:04d}_*.pdb"))
        assert cands, f"rank {rank} PDB 缺失"
        f = cands[0]
        jobs.append((f"mpnn#{rank}", str(f), r["binder_sequence"], stage_row,
                     r["iptm"], r["distogram_iptm_proxy"], r["isoelectric_point"]))
    s1_row = s1.iloc[0]
    s1_pdb = base / "stage1_only" / f"stage1_{s1_row['hero_model']}.pdb"
    jobs.append(("stage1#1", str(s1_pdb), s1_row["binder_sequence"], "esmfold2",
                 s1_row["iptm"], s1_row["distogram_iptm_proxy"], s1_row["isoelectric_point"]))
    n_mpnn = len(jobs) - 1
    print(f"[rescore] 共 {len(jobs)} 个候选 ({n_mpnn} mpnn + 1 stage1), "
          f"链: target={target_chains} binder={binder_chain}, "
          f"{N_WORKERS} 进程并行 Rosetta (含 relax)...", flush=True)

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(_score_one, p, s, stg, binder_chain, target_chains): k
                for k, p, s, stg, *_ in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            results[k] = fut.result()
            print(f"  [{n}/{len(jobs)}] {k}: dG={results[k]['dG']:.2f} "
                  f"sc={results[k]['sc']:.2f}", flush=True)
    print(f"[rescore] Rosetta 完成, {time.time()-t0:.0f}s", flush=True)

    # ---- 重建 Stage 6 (core.py 原生逻辑) ----
    prs = []
    for key, pdb, seq, stage, iptm, proxy, pI in jobs:
        rr = results[key]
        pr = PipelineResult(
            target_name=args.task, binder_name="minibinder", stage=stage,
            binder_sequence=seq,
            refold_iptm=iptm if stage == "mpnn" else None,
            iptm=iptm if stage == "esmfold2" else None,
            refold_ptm=None, distogram_iptm_proxy=proxy,
            rosetta_dG=rr["dG"], rosetta_sc=rr["sc"],
            rosetta_hbonds=rr["hbonds"], rosetta_unsat_hbonds=rr["unsat"],
            rosetta_clashes=rr["clashes"], rosetta_binder_energy=rr["binder_energy"],
            isoelectric_point=pI if pI is not None else _compute_pI(seq),
        )
        pr._key = key
        pr._pdb = pdb
        prs.append(pr)

    all_scored = compute_selection_score(prs, rosetta_weight=ROSETTA_WEIGHT)
    passed, failed = apply_filters(all_scored)
    selected = select_top_n(passed, top_n=84)
    selected_keys = {p._key for p in selected}

    # ---- 重写产物 ----
    def fmt(v, w=8, prec=4):
        return f"{v:>{w}.{prec}f}" if v is not None else " " * (w - 2) + "NA"
    lines = [
        f"# e2e 设计结果 (Stage 5/6 离线重评分, Rosetta weight={ROSETTA_WEIGHT})  "
        f"task={args.task}  hotspot={args.hotspot}  seed={args.seed}",
        f"# 生成时间 {time.strftime('%F %T')}  |  候选 {len(jobs)} = 1 stage1 + "
        f"{n_mpnn} mpnn, 全部参与打分排序; selected 按过滤器语义产出",
        "",
        f"{'rank':>4}  {'stage':<9}  {'iptm':>8}  {'dG':>9}  {'score':>8}  "
        f"{'filter':<6}  {'sel':>3}  binder_sequence",
    ]
    rows = []
    for rank, r in enumerate(all_scored, 1):
        lines.append(
            f"{rank:>4}  {r.stage:<9}  {fmt(r.refold_iptm or r.iptm)}  "
            f"{fmt(r.rosetta_dG, 9, 2)}  {fmt(r.selection_score)}  "
            f"{'PASS' if r.filter_passed else 'FAIL':<6}  "
            f"{'Y' if r._key in selected_keys else 'N':>3}  {r.binder_sequence}")
        rows.append({
            "rank": rank, "key": r._key, "stage": r.stage,
            "binder_sequence": r.binder_sequence, "iptm": r.refold_iptm or r.iptm,
            "rosetta_dG": r.rosetta_dG, "rosetta_sc": r.rosetta_sc,
            "rosetta_hbonds": r.rosetta_hbonds,
            "rosetta_unsat": r.rosetta_unsat_hbonds,
            "rosetta_clashes": r.rosetta_clashes,
            "rosetta_binder_energy": r.rosetta_binder_energy,
            "distogram_iptm_proxy": r.distogram_iptm_proxy,
            "isoelectric_point": r.isoelectric_point,
            "selection_score": r.selection_score,
            "filter_passed": r.filter_passed, "filter_unmet": r.filter_unmet,
            "selected": r._key in selected_keys, "pdb": r._pdb,
        })
    (base / "design_summary.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_parquet(base / "full_pipeline" / "selection_with_rosetta.parquet")

    # PDB 注入 REMARK dG (幂等: 剔除旧 rosetta_dG 行后重插, 防与 run_e2e 重复)
    for r in prs:
        rr = results[r._key]
        p = Path(r._pdb)
        txt_lines = [ln for ln in p.read_text().split("\n")
                     if "rosetta_dG=" not in ln]
        ins = None
        for i, ln in enumerate(txt_lines):
            if ln.startswith("REMARK 250"):
                ins = i + 1
            elif ln.startswith(("ATOM", "HETATM")):
                break
        txt_lines.insert(ins if ins is not None else 0,
                         f"REMARK 250  rosetta_dG={rr['dG']:.4f}")
        p.write_text("\n".join(txt_lines))

    n_pass = sum(r.filter_passed for r in prs)
    print(f"\n[rescore] 完成: {len(jobs)} 候选打分排序, filter 通过 {n_pass}, "
          f"selected {len(selected)} (含 stage1: {'stage1#1' in selected_keys})")
    print(f"[rescore] Top-3:")
    for rank, r in enumerate(all_scored[:3], 1):
        print(f"  {rank}. [{r.stage}] iPTM={fmt(r.refold_iptm or r.iptm)} "
              f"dG={fmt(r.rosetta_dG, 9, 2)} score={fmt(r.selection_score)}")
    print(f"[rescore] 产物: design_summary.txt / selection_with_rosetta.parquet / PDB REMARK")


if __name__ == "__main__":
    main()
