#!/usr/bin/env python3
"""
诊断实验: binder 长度对照 (158 aa vs 94 aa) + 脱域误判检验。

对照事实: 4ZQK/无热点 最佳 binder 158 aa (iPTM 0.91); 4ZQK/热点 seed1
minibinder 采样出 94 aa, founder iPTM 0.8245 / dG-47.67 但后代弱。
本脚本以 158 aa 全可变模板 ("#"*158) 重跑 Stage1+质量门, 并持久化
结构/逐残基 pLDDT, 检验: 低 iPTM 是否因"部分结构域脱离界面"造成的
质量误判 (真 pLDDT 高 + 界面接触存在 = 支持误判说; pLDDT 全低 = 否定)。

用法: W_EPI=0.5|0.05 CUDA_VISIBLE_DEVICES=<UUID> python diag_len158.py
"""
from __future__ import annotations

import json
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

import run_e2e  # noqa: E402,F401  (Patch A + pyrosetta.init + hero 限 1)
import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import ESMFold2Design  # noqa: E402

W_EPI = float(os.environ.get("W_EPI", "0.05"))
bd.LOSS_WEIGHTS["epitope"] = W_EPI

BINDER_LEN = 158
SEED = 1
HOTSPOTS = ["B:S30", "A:S100"]

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def plddt_profile(pdb_text: str) -> dict:
    """B-factor 列 = pLDDT (AlphaFold 约定)。按链统计。"""
    per = {}
    for ln in pdb_text.split("\n"):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ch = ln[21]
            per.setdefault(ch, []).append(float(ln[60:66]))
    return {ch: {"n": len(v), "mean": sum(v) / len(v), "min": min(v)}
            for ch, v in per.items()}


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
    template = "#" * BINDER_LEN
    print(f"[len158] 模型加载 {time.time()-t0:.0f}s; binder={BINDER_LEN} aa, "
          f"w_epi={W_EPI}, steps=150, seed={SEED}", flush=True)

    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name=None,
        binder_sequence=template,
        target_sequence=target_seq, is_antibody=False,
        seed=SEED, batch_size=1,
        target_hotspot_ids=HOTSPOTS, epitope_contact_distance=12.0,
    )
    print(f"[len158] Stage-1 完成 {time.time()-t0:.0f}s", flush=True)

    from core import IntegratedPipeline, PipelineConfig
    cfg = PipelineConfig(verbose=True)
    pipeline = IntegratedPipeline(cfg)
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    entry = (quality_passed or quality_failed)[0]
    gate = "PASS" if quality_passed else "FAIL"
    iptm, ptm = entry.get("iptm"), entry.get("ptm")
    proxy = entry.get("distogram_iptm_proxy")

    out = _pipeline_dir / "test" / "module_results" / "diag_len158" / f"w{W_EPI}"
    out.mkdir(parents=True, exist_ok=True)
    pdb_text = entry.get("annotated_pdb") or ""
    (out / "design.pdb").write_text(pdb_text)
    prof = plddt_profile(pdb_text) if pdb_text else {}
    print(f"\n[len158] 质量门: {gate} | iPTM={iptm} pTM={ptm} "
          f"proxy={proxy} (94aa 基线 iPTM=0.8245)", flush=True)
    for ch, s in sorted(prof.items()):
        tag = "binder" if ch == "C" else "target"
        print(f"[len158] pLDDT 链{ch}({tag}): 均值 {s['mean']:.1f} "
              f"min {s['min']:.1f} ({s['n']} 残基)", flush=True)

    (out / "result.json").write_text(json.dumps({
        "binder_len": BINDER_LEN, "seed": SEED, "hotspots": HOTSPOTS,
        "epitope_weight": W_EPI, "steps": 150,
        "gate": gate, "iptm": iptm, "ptm": ptm, "proxy": proxy,
        "plddt_profile": prof,
        "designed_sequence": entry.get("designed_sequence", ""),
    }, indent=2))
    print(f"[len158] 产物 → {out}", flush=True)


if __name__ == "__main__":
    main()
