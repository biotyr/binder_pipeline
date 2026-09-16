#!/usr/bin/env python3
"""
T7: Stage 5 Rosetta 物理评分 — screen_single_pdb 真跑 (PyRosetta, 纯 CPU)。

1. 单链/多链伪复合物 × relax_first=False (快路径: 冲突检测 + InterfaceAnalyzer
   + DSSP 二级结构 + 序列检查)
2. 单链 × relax_first=True (完整路径, FastRelax 较慢)

与 BindCraft 原版一致, pr.init 需带 -holes:dalphaball (BuriedUnsatHbonds
filter 的 dalphaball_sasa=1 依赖)。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import pyrosetta as pr  # noqa: E402

DALPHABALL = "/home/taoyr/projects/BindCraft-main/functions/DAlphaBall.gcc"
DSSP = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"

pr.init(f"-holes:dalphaball {DALPHABALL} -mute all")

from bindercraft_filter.integration import screen_single_pdb  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = _pipeline_dir / "test" / "module_results"
RESULTS.mkdir(exist_ok=True)


def run_case(tag: str, pdb: str, binder_chain: str, target_chains: list[str],
             relax_first: bool) -> dict:
    manifest = json.loads((HERE / "sim_manifest.json").read_text())[pdb]
    t0 = time.time()
    print(f"\n[{tag}] screen_single_pdb({pdb}, binder={binder_chain}, "
          f"targets={target_chains}, relax_first={relax_first})", flush=True)
    res = screen_single_pdb(
        pdb_path=str(HERE / "sim_pdbs" / pdb),
        binder_sequence=manifest["binder_sequence"],
        dssp_path=DSSP,
        binder_chain=binder_chain,
        target_chains=target_chains,
        relax_first=relax_first,
        relaxed_output_dir=str(HERE / "relaxed"),
        omit_AAs="C",
    )
    elapsed = time.time() - t0
    iface = res["interface_scores"]
    print(f"[{tag}] {elapsed:.0f}s  dG={iface['interface_dG']}  "
          f"sc={iface['interface_sc']}  packstat={iface['interface_packstat']}  "
          f"hbonds={iface['interface_interface_hbonds']}  "
          f"unsat={iface['interface_delta_unsat_hbonds']}  "
          f"nres={iface['interface_nres']}")
    print(f"[{tag}] clashes before/after relax: "
          f"{res['clashes_before_relax']}/{res['clashes_after_relax']}  "
          f"ss: {res['ss_percentages']}")
    print(f"[{tag}] sequence_notes: {res['sequence_notes']!r}")
    return {
        "elapsed_s": round(elapsed, 1),
        "relax_first": relax_first,
        "interface_scores": iface,
        "interface_AA": res["interface_AA"],
        "clashes": {"before": res["clashes_before_relax"],
                    "after": res["clashes_after_relax"],
                    "ca": res["ca_clashes"]},
        "ss_percentages": res["ss_percentages"],
        "i_pLDDT": res["i_pLDDT"], "ss_pLDDT": res["ss_pLDDT"],
        "sequence_notes": res["sequence_notes"],
        "relaxed_pdb": res.get("relaxed_pdb"),
    }


def main() -> None:
    report = {}
    report["single_fast"] = run_case(
        "T7a", "single_pass.pdb", "B", ["A"], relax_first=False)
    report["multi_fast"] = run_case(
        "T7b", "multi_pass.pdb", "C", ["A", "B"], relax_first=False)
    report["single_relax"] = run_case(
        "T7c", "single_pass.pdb", "B", ["A"], relax_first=True)

    (RESULTS / "t7_rosetta.json").write_text(json.dumps(report, indent=2))
    print(f"\n[T7] done -> {RESULTS / 't7_rosetta.json'}")


if __name__ == "__main__":
    main()
