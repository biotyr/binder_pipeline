#!/usr/bin/env python3
"""
4ZQK 毒点三角定位实验 (Stage1+质量门): 只换热点对, 其余 = 任务#2 崩溃配置
(minibinder 工厂 seed0 → 158aa, w=0.5, 150 步)。
  i_BS30_AE28 : B:S30(界面边缘) + A:E28(非界面)   — 间距 22.9Å
  ii_AQ30_AD73: A:Q30 + A:D73 (双非界面, 同链 A)   — 间距 14.6Å
判读: PASS/FAIL × 间距 → 定位毒因 (边缘锚定 vs 间距 vs 双重边缘钉扎)。
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
for p in (str(_pipeline_dir), str(_pipeline_dir / "test" / "e2e")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Bio import SeqIO  # noqa: E402

import run_e2e  # noqa: E402,F401
import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import ESMFold2Design  # noqa: E402

bd.LOSS_WEIGHTS["epitope"] = 0.5

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seq_health(seq: str) -> dict:
    cnt = collections.Counter(seq)
    H = -sum((c / len(seq)) * math.log2(c / len(seq)) for c in cnt.values())
    maxrun, cur = 1, 1
    for a, b in zip(seq, seq[1:]):
        cur = cur + 1 if a == b else 1
        maxrun = max(maxrun, cur)
    top2 = sum(c for _, c in cnt.most_common(2)) / len(seq)
    return {"entropy": round(H, 2), "max_homopolymer": maxrun,
            "top2_aa_frac": round(top2, 2)}


def run_one(tag: str, hotspots: list[str], app, target_seq: str) -> dict:
    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name="minibinder",
        target_sequence=target_seq, is_antibody=False,
        seed=0, batch_size=1,
        target_hotspot_ids=hotspots, epitope_contact_distance=12.0,
    )
    from core import IntegratedPipeline, PipelineConfig
    pipeline = IntegratedPipeline(PipelineConfig(verbose=True))
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    entry = (quality_passed or quality_failed)[0]
    gate = "PASS" if quality_passed else "FAIL"
    binder_seq = (entry.get("designed_sequence") or "").split("|")[-1]
    health = seq_health(binder_seq)
    res = {"tag": tag, "hotspots": hotspots, "gate": gate,
           "iptm": entry.get("iptm"), "ptm": entry.get("ptm"),
           "proxy": entry.get("distogram_iptm_proxy"),
           "seq_health": health, "binder_sequence": binder_seq,
           "elapsed_s": round(time.time() - t0, 1)}
    print(f"\n[mixed] {tag}: 门={gate} iPTM={res['iptm']} pTM={res['ptm']} "
          f"proxy={res['proxy']} 健康={health}", flush=True)
    print(f"[mixed] binder: {binder_seq[:60]}...", flush=True)
    out = (_pipeline_dir / "test" / "module_results" / "diag_4zqk_mixed" / tag)
    out.mkdir(parents=True, exist_ok=True)
    (entry.get("annotated_pdb") and
        (out / "design.pdb").write_text(entry["annotated_pdb"]))
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    DATA = _pipeline_dir / "test" / "data"
    seqs = {}
    for r in SeqIO.parse(DATA / "rcsb_pdb_4ZQK.fasta", "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        seqs[m.group(1)] = str(r.seq).replace("X", "")
    target_seq = f"{seqs['A']}|{seqs['B']}"

    app = ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[mixed] 模型加载 {time.time()-t0:.0f}s; w=0.5, factory(158), seed0",
          flush=True)

    run_one("i_BS30_AE28", ["B:S30", "A:E28"], app, target_seq)
    run_one("ii_AQ30_AD73", ["A:Q30", "A:D73"], app, target_seq)
    print("[mixed] 两轮完成", flush=True)


if __name__ == "__main__":
    main()
