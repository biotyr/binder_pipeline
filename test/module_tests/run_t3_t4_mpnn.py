#!/usr/bin/env python3
"""
T3+T4: Stage 3 ProteinMPNN 逆折叠重设计 (单链 / 多链 target)。

走 core.py Stage 3 的生产路径 mpnn_redesign_from_critic_results:
从模拟 critic_results (annotated_pdb + designed_sequence) 出发,
链名自动推导 -> 界面检测 (cKDTree 4Å) -> 固定 target 全链 + binder
界面残基 -> MPNN 采样 20 条。

JAX_PLATFORMS=cpu: MPNN 很小, 不占用 GPU。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from inverse_optimize.mpnn_redesign import (  # noqa: E402
    mpnn_redesign_from_critic_results,
)

HERE = Path(__file__).resolve().parent
RESULTS = _pipeline_dir / "test" / "module_results"
RESULTS.mkdir(exist_ok=True)


def fake_critic_result(pdb_name: str) -> dict:
    """模拟 critic_results 条目 — key 与 Stage 1 Hero 输出一致。"""
    manifest = json.loads((HERE / "sim_manifest.json").read_text())[pdb_name]
    pdb_text = (HERE / "sim_pdbs" / pdb_name).read_text()
    from esm.utils.structure.protein_complex import ProteinComplex

    return {
        "critic_name": f"sim::{pdb_name}",
        "is_scaling_critic": False,
        "designed_sequence": manifest["designed_sequence"],
        "annotated_pdb": pdb_text,
        "complex": ProteinComplex.from_pdb(str(HERE / "sim_pdbs" / pdb_name)),
        "iptm": manifest.get("iptm"),
    }


def main() -> None:
    report = {}
    cases = [
        ("T3_single", "single_pass.pdb", 20),
        ("T4_multi", "multi_pass.pdb", 20),
    ]
    for tag, pdb_name, n_seqs in cases:
        print(f"\n{'='*60}\n[{tag}] MPNN redesign of {pdb_name}\n{'='*60}")
        t0 = time.time()
        out = mpnn_redesign_from_critic_results(
            [fake_critic_result(pdb_name)],
            num_seqs=n_seqs,
            sampling_temp=0.1,
            fix_interface=True,
            random_seed=0,
        )
        elapsed = time.time() - t0
        entry = out[0]
        designs = entry["mpnn_designs"]
        assert len(designs) == n_seqs, f"{tag}: 仅 {len(designs)} 条序列"
        # 序列一致性检查: 所有 MPNN 序列长度 == binder 长度, 且不含 C (omit_AAs)
        manifest = json.loads((HERE / "sim_manifest.json").read_text())[pdb_name]
        blen = len(manifest["binder_sequence"])
        for d in designs:
            assert len(d["sequence"]) == blen, (
                f"{tag}: 长度 {len(d['sequence'])} != binder {blen}")
            assert "C" not in d["sequence"], f"{tag}: 含禁用氨基酸 C"
        scores = [d["score"] for d in designs]
        seqids = [d["seqid"] for d in designs]
        report[tag] = {
            "pdb": pdb_name,
            "critic": entry["critic_name"],
            "binder_len": blen,
            "n_designs": len(designs),
            "score_range": [min(scores), max(scores)],
            "seqid_range": [min(seqids), max(seqids)],
            "top1_sequence": designs[0]["sequence"],
            "top1_score": designs[0]["score"],
            "elapsed_s": round(elapsed, 1),
            "designs": designs,
        }
        print(f"[{tag}] {len(designs)} seqs, len={blen}, "
              f"score [{min(scores):.3f}, {max(scores):.3f}], "
              f"seqid [{min(seqids):.3f}, {max(seqids):.3f}], "
              f"{elapsed:.0f}s")
        print(f"[{tag}] top1: {designs[0]['sequence']}")

    (RESULTS / "t3_t4_mpnn.json").write_text(json.dumps(report, indent=2))
    print(f"\n[T3+T4] done -> {RESULTS / 't3_t4_mpnn.json'}")


if __name__ == "__main__":
    main()
