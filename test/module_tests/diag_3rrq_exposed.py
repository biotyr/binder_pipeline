#!/usr/bin/env python3
"""
判别实验: 3RRQ 暴露表面热点 (对照原凹槽热点的成功)。

假设检验: 4ZQK 的挫折是否为"暴露表位在 w=0.5 下可达退化解/包裹解"的特例?
配置完全复刻任务1' (3RRQ/凹槽热点, 成功: iPTM 0.8575) — minibinder 工厂
(seed0 → 158 aa), w=0.5, 150 步, seed0 — 仅热点换成暴露对 PDB A:S60+A:P130
(fasta S29/P99, 9.1Å, 两链最暴露 12% 残基, 远离凹槽)。
只跑 Stage1+质量门; 序列健康度 (熵/重复) 一并量化, 对照包裹型签名
(熵 2.77 / S+R 62% / 4ZQK 包裹解)。
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

bd.LOSS_WEIGHTS["epitope"] = 0.5   # 与任务1'一致

HOTSPOTS = ["A:S29", "A:P99"]
SEED = 0


def seq_health(seq: str) -> dict:
    cnt = collections.Counter(seq)
    H = -sum((c/len(seq)) * math.log2(c/len(seq)) for c in cnt.values())
    maxrun, cur = 1, 1
    for a, b in zip(seq, seq[1:]):
        cur = cur + 1 if a == b else 1
        maxrun = max(maxrun, cur)
    top2 = sum(c for _, c in cnt.most_common(2)) / len(seq)
    return {"entropy": round(H, 2), "max_homopolymer": maxrun,
            "top2_aa_frac": round(top2, 2)}


def main() -> None:
    DATA = _pipeline_dir / "test" / "data"
    fa = str(next(SeqIO.parse(DATA / "rcsb_pdb_3RRQ.fasta", "fasta")).seq).replace("X", "")

    app = bd.ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[exposed] 模型加载 {time.time()-t0:.0f}s; 3RRQ 单链, 热点={HOTSPOTS}, "
          f"w=0.5, steps=150, seed=0, minibinder 工厂", flush=True)

    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="3RRQ_single", binder_name="minibinder",
        target_sequence=fa, is_antibody=False,
        seed=SEED, batch_size=1,
        target_hotspot_ids=HOTSPOTS, epitope_contact_distance=12.0,
    )
    print(f"[exposed] Stage-1 完成 {time.time()-t0:.0f}s", flush=True)

    from core import IntegratedPipeline, PipelineConfig
    pipeline = IntegratedPipeline(PipelineConfig(verbose=True))
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    entry = (quality_passed or quality_failed)[0]
    gate = "PASS" if quality_passed else "FAIL"
    iptm, ptm = entry.get("iptm"), entry.get("ptm")
    proxy = entry.get("distogram_iptm_proxy")

    binder_seq = (entry.get("designed_sequence") or "").split("|")[-1]
    health = seq_health(binder_seq)
    print(f"\n[exposed] 质量门: {gate} | iPTM={iptm} pTM={ptm} proxy={proxy} "
          f"(凹槽基线 iPTM=0.8575)", flush=True)
    print(f"[exposed] 序列健康: {health} (包裹型签名: 熵≈2.77, top2≈62%)",
          flush=True)
    print(f"[exposed] binder: {binder_seq[:60]}...", flush=True)

    out = _pipeline_dir / "test" / "module_results" / "diag_3rrq_exposed"
    out.mkdir(parents=True, exist_ok=True)
    pdb_text = entry.get("annotated_pdb") or ""
    (out / "design.pdb").write_text(pdb_text)
    (out / "result.json").write_text(json.dumps({
        "hotspots": HOTSPOTS, "epitope_weight": 0.5, "steps": 150, "seed": SEED,
        "gate": gate, "iptm": iptm, "ptm": ptm, "proxy": proxy,
        "seq_health": health, "binder_sequence": binder_seq,
        "designed_sequence": entry.get("designed_sequence", ""),
    }, indent=2))
    print(f"[exposed] 产物 → {out}", flush=True)


if __name__ == "__main__":
    main()
