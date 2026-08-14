#!/usr/bin/env python3
# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm@git+https://github.com/Biohub/esm.git@main",
#     "modal",
#     "pandas",
#     "biopython",
#     "scipy",
#     "numpy",
# ]
# ///
"""
Modal 部署入口 — IntegratedPipeline 的远程 GPU 适配器。

部署
════
  modal deploy run_modal.py

运行
════
  modal run run_modal.py --target-name pd-l1 --binder-name minibinder
  modal run run_modal.py --target-name pd-l1 --binder-name minibinder --enable-mpnn --enable-rosetta

核心逻辑位于 ``core.py``，本地运行见 ``run_local.py``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import modal

# ── 确保 pipeline/ 包可被导入 ──────────────────────────────────────
_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from core import IntegratedPipeline, PipelineConfig  # noqa: E402

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


# ── Modal 镜像 ──────────────────────────────────────────────────────


def _get_modal_image() -> Any:
    """构建包含所有 Pipeline 依赖的 Modal 镜像。"""
    return (
        modal.Image.micromamba(python_version="3.12")
        .run_commands("apt update && apt install -y git build-essential")
        .micromamba_install(
            "anarci>=2020.04.03",
            "hmmer=3.4",
            "cuda-version=12.8",
            "cuda-libraries-dev=12.8",
            "cuda-nvcc=12.8",
            "cmake",
            "ninja",
            channels=["conda-forge", "bioconda"],
        )
        .pip_install(
            "torch==2.8.0",
            "triton==3.4.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "flash-attn==2.8.3",
            "transformer-engine[core-cu12,pytorch]==2.13.0",
            "xformers==0.0.32.post1",
            extra_options="--no-build-isolation",
            env={
                "CPATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/include"
                ),
                "LIBRARY_PATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/lib"
                ),
                "MAX_JOBS": "8",
                "NVTE_FRAMEWORK": "pytorch",
            },
        )
        .pip_install(
            "abnumber",
            "esm@git+https://github.com/Biohub/esm.git@main",
            "modal",
            "biopython",
            "scipy",
            "pandas",
            "pyarrow",
        )
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


app = modal.App(
    name="integrated-binder-pipeline",
    image=_get_modal_image(),
    volumes={
        "/models": modal.Volume.from_name(
            "esmfold2-models", create_if_missing=True
        )
    },
)


@app.cls(gpu="H100", timeout=120 * 60, cpu=16, memory=80 * 1024)
class IntegratedPipelineModal(IntegratedPipeline):
    """Modal 远端入口 — GPU 上执行完整 IntegratedPipeline.run()。"""

    @modal.enter()
    def load_models(self):
        """预加载 ESMFold2 模型 (在容器启动时执行一次)。"""
        pass

    @modal.method()
    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().run(*args, **kwargs)


@app.local_entrypoint()
def main(
    target_name: str = "pd-l1",
    binder_name: str = "minibinder",
    target_sequence: str | None = None,
    binder_sequence: str | None = None,
    is_antibody: bool | None = None,
    seed: int = 0,
    batch_size: int = 1,
    target_hotspot_ids: list[str] | None = None,
    epitope_contact_distance: float = 12.0,
    enable_mpnn: bool = True,
    enable_rosetta: bool = False,
    mpnn_num_seqs: int = 20,
    hero_critic_names: list[str] | None = None,
    top_n: int = 84,
    output_dir: str = "./pipeline_output",
    local: bool = False,
):
    """CLI 入口: 可通过 ``modal run`` 或本地 ``python run_modal.py --local`` 调用。

    示例::

        modal run run_modal.py --target-name pd-l1 --binder-name minibinder
        modal run run_modal.py --target-name egfr --binder-name minibinder --enable-mpnn
    """
    config = PipelineConfig(
        enable_mpnn=enable_mpnn,
        enable_rosetta=enable_rosetta,
        mpnn_num_seqs=mpnn_num_seqs,
        hero_critic_names=(
            hero_critic_names
            if hero_critic_names is not None
            else ["ESMFold2-Experimental-Cutoff2025"]
        ),
        top_n=top_n,
        output_dir=output_dir,
    )

    if local:
        pipeline = IntegratedPipeline(config)
        pipeline_fn = pipeline.run
    else:
        pipeline = IntegratedPipelineModal()
        pipeline_fn = pipeline.run.remote

    result = pipeline_fn(
        target_name=target_name,
        binder_name=binder_name,
        target_sequence=target_sequence,
        binder_sequence=binder_sequence,
        is_antibody=is_antibody,
        seed=seed,
        batch_size=batch_size,
        target_hotspot_ids=target_hotspot_ids,
        epitope_contact_distance=epitope_contact_distance,
    )

    # 打印摘要
    selected = result.get("selected", [])
    if isinstance(selected, list) and selected:
        print(f"\n{'='*60}")
        print(f"Top-{min(5, len(selected))} Designs")
        print(f"{'='*60}")
        for i, r in enumerate(selected[:5], 1):
            seq_short = r.binder_sequence[:40] + (
                "..." if len(r.binder_sequence) > 40 else ""
            )
            print(
                f"  {i}. [{r.stage:8s}] "
                f"score={r.selection_score:.4f}  "
                f"iPTM={r.refold_iptm or r.iptm}  "
                f"{seq_short}"
            )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Binder Design Pipeline")
    parser.add_argument("--target-name", default="pd-l1")
    parser.add_argument("--binder-name", default="minibinder")
    parser.add_argument("--target-sequence", default=None)
    parser.add_argument("--binder-sequence", default=None)
    parser.add_argument("--is-antibody", type=bool, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--enable-mpnn", type=bool, default=True)
    parser.add_argument("--enable-rosetta", type=bool, default=False)
    parser.add_argument("--mpnn-num-seqs", type=int, default=20)
    parser.add_argument(
        "--hero-critics", nargs="*", default=None,
        help="Hero critics for refold screening, default: ESMFold2-Experimental-Cutoff2025",
    )
    parser.add_argument("--top-n", type=int, default=84)
    parser.add_argument("--output-dir", default="./pipeline_output")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    main(
        target_name=args.target_name,
        binder_name=args.binder_name,
        target_sequence=args.target_sequence,
        binder_sequence=args.binder_sequence,
        is_antibody=args.is_antibody,
        seed=args.seed,
        batch_size=args.batch_size,
        enable_mpnn=args.enable_mpnn,
        enable_rosetta=args.enable_rosetta,
        mpnn_num_seqs=args.mpnn_num_seqs,
        hero_critic_names=args.hero_critics,
        top_n=args.top_n,
        output_dir=args.output_dir,
        local=args.local,
    )
