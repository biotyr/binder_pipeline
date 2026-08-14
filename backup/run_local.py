#!/usr/bin/env python3
# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm@git+https://github.com/Biohub/esm.git@main",
#     "pandas",
#     "biopython",
#     "scipy",
#     "numpy",
#     "torch>=2.8.0",
# ]
# ///
"""
本地运行入口 — 无需 Modal, 直接在本地 GPU 上执行完整 Pipeline。

使用
════
  python run_local.py --target-name pd-l1 --binder-name minibinder
  python run_local.py --target-name egfr --binder-name minibinder \\
      --enable-mpnn --mpnn-num-seqs 30 --top-n 20

依赖
════
  见 ``run_modal.py`` 中 ``_get_modal_image()`` 的安装列表。
  核心: torch≥2.8.0 + triton + flash-attn + transformer-engine + xformers
       + esm (Biohub) + biopython + scipy + pandas
  可选: pyrosetta (Rosetta 评分, 商业许可)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── 确保 pipeline/ 包可被导入 ──────────────────────────────────────
_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from core import IntegratedPipeline, PipelineConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated Binder Design Pipeline (local)"
    )
    parser.add_argument("--target-name", default="pd-l1")
    parser.add_argument("--binder-name", default="minibinder")
    parser.add_argument("--target-sequence", default=None)
    parser.add_argument("--binder-sequence", default=None)
    parser.add_argument("--is-antibody", type=bool, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--target-hotspot-ids", nargs="*", default=None,
        help="1-indexed target hotspot residues, e.g. L150 Y123",
    )
    parser.add_argument(
        "--epitope-contact-distance", type=float, default=12.0,
    )
    parser.add_argument("--enable-mpnn", type=bool, default=True)
    parser.add_argument("--enable-rosetta", type=bool, default=False)
    parser.add_argument("--mpnn-num-seqs", type=int, default=20)
    parser.add_argument("--mpnn-sampling-temp", type=float, default=0.1)
    parser.add_argument("--quality-min-iptm", type=float, default=0.50)
    parser.add_argument("--scaling-threshold", type=float, default=0.50)
    parser.add_argument(
        "--hero-critics", nargs="*", default=None,
        help=(
            "Hero critics for refold screening, e.g. "
            "ESMFold2-Experimental-Cutoff2025 ESMFold2-Experimental. "
            "Default: ESMFold2-Experimental-Cutoff2025"
        ),
    )
    parser.add_argument("--top-n", type=int, default=84)
    parser.add_argument("--output-dir", default="./pipeline_output")
    parser.add_argument("--no-save-pdbs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        enable_mpnn=args.enable_mpnn,
        enable_rosetta=args.enable_rosetta,
        mpnn_num_seqs=args.mpnn_num_seqs,
        mpnn_sampling_temp=args.mpnn_sampling_temp,
        quality_min_iptm=args.quality_min_iptm,
        scaling_threshold=args.scaling_threshold,
        hero_critic_names=(
            args.hero_critics
            if args.hero_critics is not None
            else ["ESMFold2-Experimental-Cutoff2025"]
        ),
        top_n=args.top_n,
        output_dir=args.output_dir,
        save_pdbs=not args.no_save_pdbs,
        verbose=not args.quiet,
    )

    pipeline = IntegratedPipeline(config)
    result = pipeline.run(
        target_name=args.target_name,
        binder_name=args.binder_name,
        target_sequence=args.target_sequence,
        binder_sequence=args.binder_sequence,
        is_antibody=args.is_antibody,
        seed=args.seed,
        batch_size=args.batch_size,
        target_hotspot_ids=args.target_hotspot_ids,
        epitope_contact_distance=args.epitope_contact_distance,
    )

    selected = result.get("selected", [])
    if isinstance(selected, list) and selected:
        print(f"\n{'='*60}")
        print(f"Top-{min(10, len(selected))} Designs")
        print(f"{'='*60}")
        for i, r in enumerate(selected[:10], 1):
            seq_short = r.binder_sequence[:50] + (
                "..." if len(r.binder_sequence) > 50 else ""
            )
            print(
                f"  {i:2d}. [{r.stage:8s}] "
                f"score={r.selection_score:.4f}  "
                f"iPTM={r.refold_iptm or r.iptm}  "
                f"{seq_short}"
            )
    else:
        print("No designs selected.")

    elapsed = result.get("elapsed_s", 0)
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
