#!/usr/bin/env python3
"""
T5: Stage 4 Scaling 快筛 — 用轻量 critic (Fast-base300M-step1500k,
~3GB 显存) 对 T3 的 MPNN 序列跑真·1 步扩散 distogram 筛选。
   (Hero 精测的 6B 模型需 ~24GB+, 留待空卡阶段)

T6: epitope-loss 数值路径 — 同一 critic 的真实 forward distogram 上,
对 3RRQ/4ZQK 热点跑 compute_structure_losses 带/不带 target_hotspot_ids,
验证 Stage 1 设计循环中热点损失代码在 GPU 上真实可跑。

运行: CUDA_VISIBLE_DEVICES=3 (避开同事任务占用最多的 GPU 0)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from Bio import SeqIO  # noqa: E402

from esmcp.binder_design import (  # noqa: E402
    _load_hf_model,
    compute_structure_losses,
)
from refold.screen import RefoldScreen, RefoldScreenConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = _pipeline_dir / "test" / "module_results"
DATA = _pipeline_dir / "test" / "data"
SCALING_CRITIC = "ESMFold2-Experimental-Fast-base300M-step1500k"
N_SEQS = 8  # 快筛序列数 (每条一次前向)


def fasta_chains(p: Path) -> dict[str, str]:
    out = {}
    for r in SeqIO.parse(p, "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        out[m.group(1)] = str(r.seq).replace("X", "")
    return out


def main() -> None:
    t0 = time.time()
    print(f"[load] {SCALING_CRITIC} (cache_esmc=True)...")
    model = _load_hf_model(
        SCALING_CRITIC, lm_dropout=0.0, cache_esmc=True, device="cuda"
    )
    print(f"[load] done in {time.time()-t0:.0f}s, "
          f"VRAM={torch.cuda.memory_allocated()/2**30:.2f} GiB")

    mpnn = json.loads((RESULTS / "t3_t4_mpnn.json").read_text())
    seqs = [d["sequence"] for d in mpnn["T3_single"]["designs"][:N_SEQS]]
    manifest = json.loads((HERE / "sim_manifest.json").read_text())
    single_target = manifest["single_pass.pdb"]["designed_sequence"].split("|")[0]

    # ---------- T5: scaling_filter ----------
    print(f"\n{'='*60}\n[T5] scaling filter: {len(seqs)} MPNN seqs, "
          f"target={single_target[:20]}...\n{'='*60}")
    screen = RefoldScreen(
        target_seq=single_target,
        scaling_models={SCALING_CRITIC: model},
        hero_models={},  # Hero 需 6B, 本阶段不加载
        config=RefoldScreenConfig(scaling_threshold=0.50, is_antibody=False),
        seed=0,
    )
    passed = screen.scaling_filter(seqs, threshold=0.50)
    t5 = {
        "critic": SCALING_CRITIC,
        "n_seqs": len(seqs),
        "threshold": 0.50,
        "n_passed": len(passed),
        "scores": sorted(
            (p["distogram_iptm_proxy"] for p in passed), reverse=True
        ),
        "passed": [
            {k: p[k] for k in ("sequence", "distogram_iptm_proxy")}
            for p in passed
        ],
    }
    print(f"[T5] {len(passed)}/{len(seqs)} passed, "
          f"scores={[f'{s:.3f}' for s in t5['scores']]}")

    # ---------- T6: epitope loss 路径 ----------
    print(f"\n{'='*60}\n[T6] epitope-loss path on real distogram\n{'='*60}")
    f3 = fasta_chains(DATA / "rcsb_pdb_3RRQ.fasta")
    f4 = fasta_chains(DATA / "rcsb_pdb_4ZQK.fasta")
    # 解析 hotspot.txt (pdb id 行的下一行为热点设置)
    lines = (DATA / "hotspot.txt").read_text().split("\n")
    hotspots = {lines[i].strip(): lines[i + 1].split() for i in range(0, len(lines) - 1, 2)}

    from refold.refolder import refold_distogram as _refold

    t6_cases = []
    for tag, target_seq, hs_ids in [
        ("3RRQ_single_with_hs", f3["A"], hotspots["3RRQ"]),
        ("3RRQ_single_no_hs", f3["A"], None),
        ("4ZQK_multi_with_hs", f4["A"] + "|" + f4["B"], hotspots["4ZQK"]),
        ("4ZQK_multi_no_hs", f4["A"] + "|" + f4["B"], None),
    ]:
        binder_seq = seqs[0]
        fold = _refold(
            model, target_seq, binder_seq,
            num_sampling_steps=1, num_loops=1, calculate_confidence=False, seed=0,
        )
        losses = compute_structure_losses(
            fold["distogram_logits"],
            binder_length=len(binder_seq),
            target_sequence=target_seq,
            target_hotspot_ids=hs_ids,
            epitope_contact_distance=12.0,
        )
        entry = {
            "target_len": len(target_seq.replace("|", "")),
            "binder_len": len(binder_seq),
            "hotspots": hs_ids,
            "losses": {k: float(v.mean()) for k, v in losses.items()},
        }
        if hs_ids:
            assert "epitope_loss" in losses, f"{tag}: 缺 epitope_loss"
            assert torch.isfinite(losses["epitope_loss"]).all()
        else:
            assert "epitope_loss" not in losses, f"{tag}: 不应有 epitope_loss"
        t6_cases.append((tag, entry))
        print(f"[T6] {tag:22s} total={entry['losses']['total_loss']:.4f} "
              + (f"epitope={entry['losses']['epitope_loss']:.4f} "
                 if "epitope_loss" in entry["losses"] else "(no hotspot term)"))

    (RESULTS / "t5_t6_scaling.json").write_text(json.dumps(
        {"T5": t5, "T6": {k: v for k, v in t6_cases}}, indent=2))
    print(f"\n[T5+T6] done -> {RESULTS / 't5_t6_scaling.json'} "
          f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
