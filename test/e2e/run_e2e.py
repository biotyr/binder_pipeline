#!/usr/bin/env python3
"""
端到端 runner — IntegratedPipeline 完整 6 阶段 (含 Stage 5 Rosetta)。

精度方案 (与 Modal fp32 参考的偏差最小化):
  - ESMC-6B 主干: fork 内 load_esmc 原生 bfloat16() (不可改, 与参考一致)
  - 模型头: fp32 (from_pretrained 默认, 与参考一致)
  - hero critics 限 1 个 (ESMFold2-Experimental-Cutoff2025, 与 Stage-4
    默认 hero_critic_names 一致) — 显存 ~20-22GB, 单张 4090 可容
  - Stage-1 scaling critics: 已在 core.py 关闭 (诊断项, 下游零消费)

用法:
  CUDA_VISIBLE_DEVICES=<空闲卡> python test/e2e/run_e2e.py \
      --task 3RRQ_single --hotspot no
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")            # HF 被墙, 权重全在本地
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # jax(MPNN) 不预分配
# MPNN 锁 CPU 后端 (2026-09-11 实测): GPU 后端对 ProteinMPNN 自回归图的
# XLA/LLVM JIT 编译需 30-60 分钟 (llvm-worker 单核长转); CPU 后端同规模
# 20 条序列全程 ~5s (见 test/module_tests T3/T4), 且不占设计模型显存。
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import torch  # noqa: E402
from Bio import SeqIO  # noqa: E402

# ---- pyrosetta 必须在任何 pose 操作前 init (BuriedUnsatHbonds 依赖) ----
import pyrosetta as pr  # noqa: E402

DALPHABALL = "/home/taoyr/projects/BindCraft-main/functions/DAlphaBall.gcc"
DSSP = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"
pr.init(f"-holes:dalphaball {DALPHABALL} -mute all")

# ---- 精度/覆盖补丁 (不改 pipeline 源码) ----
import esmcp.binder_design as bd  # noqa: E402

# hero 限 1 个: 与 core.py hero_critic_names 默认值一致, 省掉 3 个
# fp32 hero 头 (~4.2GB)。core.py Stage-4 仍取同名 hero, 无需重载。
bd.ESMFold2Design.hero_critic_hf_paths = ["ESMFold2-Experimental-Cutoff2025"]

# Patch A (2026-09-11, 5 轮探针定位): Transition._addmm_residual 位于
# autocast 区块之外的采样后处理链 (experimental:1079 → common:2716 → 2673),
# bf16 激活 × fp32 权重在此裸撞 (compile 期 inductor pad_mm 基准测试与
# eager 均崩)。包一层 autocast bf16 让 addmm 两侧自动转换, 语义与模型
# 其余部分的 autocast 契约一致; fp32 头/bf16 主干的参考 dtype 组合不变,
# COMPILE 保持默认开启。探针: eager no_grad/grad + compile 全过 (12.7GiB)。
from transformers.models.esmfold2 import modeling_esmfold2_common as _common

_orig_addmm_residual = _common.Transition._addmm_residual

def _addmm_residual_ac(self, x, hidden):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _orig_addmm_residual(self, x, hidden)

_common.Transition._addmm_residual = _addmm_residual_ac

from core import IntegratedPipeline, PipelineConfig  # noqa: E402

DATA = _pipeline_dir / "test" / "data"
RESULTS_ROOT = _pipeline_dir / "test" / "results"

TASKS = {
    "3RRQ_single": ("rcsb_pdb_3RRQ.fasta", ["A"]),
    "4ZQK_multi": ("rcsb_pdb_4ZQK.fasta", ["A", "B"]),  # PD-L1|PD-1 顺序, 链名对应
}


def fasta_chains(p: Path) -> dict[str, str]:
    out = {}
    for r in SeqIO.parse(p, "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        out[m.group(1)] = str(r.seq).replace("X", "")
    return out


def load_target(task: str) -> tuple[str, list[str]]:
    fasta, chain_order = TASKS[task]
    chains = fasta_chains(DATA / fasta)
    # 链顺序决定 asym_id → pipeline 链名 A,B,...; 与 hotspot.txt 的链名约定一致
    return "|".join(chains[c] for c in chain_order), chain_order


def load_hotspots(task: str) -> list[str]:
    # hotspot.txt 的 key 是 PDB 名 ("3RRQ"/"4ZQK"); e2e task 名带任务后缀
    # ("3RRQ_single"/"4ZQK_multi") → 取 "_" 前缀映射 (2026/9/14 fix,
    # 此前直接用 task 名查询, with-hotspot 任务启动即 KeyError)
    lines = (DATA / "hotspot.txt").read_text().split("\n")
    pdb_key = task.split("_")[0]
    return {lines[i].strip(): lines[i + 1].split()
            for i in range(0, len(lines) - 1, 2)}[pdb_key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--hotspot", choices=["with", "no"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)  # 4090 显存约束, 用户确认回退为 1
    ap.add_argument("--mpnn-num-seqs", type=int, default=20)
    ap.add_argument("--epitope-weight", type=float, default=None,
                    help="覆盖 Stage-1 epitope loss 权重 (pipeline 默认 0.5)")
    args = ap.parse_args()

    task_dir = RESULTS_ROOT / args.task / f"{args.hotspot}_hotspot"
    full_dir = task_dir / "full_pipeline"
    s1_dir = task_dir / "stage1_only"
    for d in (full_dir, s1_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_sequence, chain_order = load_target(args.task)
    hotspots = load_hotspots(args.task) if args.hotspot == "with" else None

    # 测试脚本层覆盖 (2026/9/15, 不改 pipeline 源码): 4ZQK 双表位 (B:S30/A:S100)
    # 诊断结论 — epitope 0.5 与 inter_contact 梯度冲突致 Stage-1 崩溃 (两次独立
    # 复现, iPTM 0.446/0.463), 降权重缓和不折衷。不指定时保持 pipeline 默认 0.5。
    if args.epitope_weight is not None:
        bd.LOSS_WEIGHTS["epitope"] = args.epitope_weight

    print(f"[e2e] task={args.task} hotspot={args.hotspot} seed={args.seed} "
          f"batch_size={args.batch_size}")
    print(f"[e2e] target chains: {[(c, len(s)) for c, s in zip(chain_order, target_sequence.split('|'))]}")
    print(f"[e2e] hotspots: {hotspots}")
    print(f"[e2e] GPU: {torch.cuda.get_device_name(0)}, "
          f"free={torch.cuda.mem_get_info()[0]/2**30:.1f}GiB")

    cfg = PipelineConfig(
        enable_mpnn=True,
        enable_rosetta=True,                    # 用户规则: 测试一律启用 PyRosetta
        rosetta_dssp_path=DSSP,                 # 本机 dssp 二进制 (Biopython 默认找 mkdssp)
        rosetta_relax_first=True,
        mpnn_num_seqs=args.mpnn_num_seqs,
        hero_critic_names=["ESMFold2-Experimental-Cutoff2025"],
        top_n=84,
        output_dir=str(full_dir),
        save_pdbs=True,
        verbose=True,
    )
    pipeline = IntegratedPipeline(cfg)

    t0 = time.time()
    result = pipeline.run(
        target_name=args.task,
        binder_name="minibinder",
        target_sequence=target_sequence,
        is_antibody=None,                        # minibinder → factory 断言 False
        seed=args.seed,
        batch_size=args.batch_size,
        target_hotspot_ids=hotspots,
        epitope_contact_distance=12.0,
    )
    elapsed = time.time() - t0

    # ---- stage1_only 产物: Stage-1 hero 原始设计 (未经 MPNN/重折叠) ----
    s1_rows = []
    for r in result["final_results"]:
        if r.stage != "esmfold2":
            continue
        fname = f"stage1_{r.hero_model_name or 'hero'}.pdb"
        if r.annotated_pdb:
            (s1_dir / fname).write_text(r.annotated_pdb)
        s1_rows.append({
            "hero_model": r.hero_model_name,
            "binder_sequence": r.binder_sequence,
            "binder_len": len(r.binder_sequence),
            "iptm": r.iptm, "ptm": r.ptm,
            "distogram_iptm_proxy": r.distogram_iptm_proxy,
            "final_loss": r.final_loss,
            "isoelectric_point": r.isoelectric_point,
            "filter_passed": r.filter_passed, "filter_unmet": r.filter_unmet,
            "selection_score": r.selection_score,
            "pdb_file": fname if r.annotated_pdb else None,
        })
    import pandas as pd
    pd.DataFrame(s1_rows).to_parquet(s1_dir / "stage1_results.parquet", index=False)

    # ---- 设计结果记录: iPTM + Rosetta dG (txt 汇总 + PDB REMARK 注入) ----
    def add_dg_remark(pdb_path: Path, dg) -> None:
        """向已保存的 PDB 注入 REMARK 250 rosetta_dG 行 (iPTM 已在 REMARK 内)。"""
        if dg is None or not pdb_path.exists():
            return
        lines = pdb_path.read_text().split("\n")
        remark = f"REMARK 250  rosetta_dG={float(dg):.4f}"
        ins = None
        for i, ln in enumerate(lines):
            if ln.startswith("REMARK 250"):
                ins = i + 1          # 插到最后一条 REMARK 250 之后, 保持块连续
            elif ln.startswith(("ATOM", "HETATM")):
                break                # 无 REMARK 块时插到首条 ATOM 之前
        lines.insert(ins if ins is not None else 0, remark)
        pdb_path.write_text("\n".join(lines))

    def fmt(v, w=8, prec=4):
        return f"{v:>{w}.{prec}f}" if v is not None else " " * (w - 2) + "NA"

    txt = [
        f"# e2e 设计结果  task={args.task}  hotspot={args.hotspot}  "
        f"seed={args.seed}  batch_size={args.batch_size}  "
        f"epitope_weight={args.epitope_weight if args.epitope_weight is not None else 0.5}",
        f"# 生成时间 {time.strftime('%F %T')}  |  精度: 主干bf16(原生)+头fp32  "
        f"hero=1(Cutoff2025)  Stage1-scaling=off",
        "",
        "# [stage1_only] Stage-1 hero 原始设计 (未经 MPNN/重折叠/Rosetta, dG=NA)",
        f"{'rank':>4}  {'hero_model':<34}  {'iptm':>8}  {'dG':>8}  binder_sequence",
    ]
    for rank, r in enumerate(s1_rows, 1):
        txt.append(f"{rank:>4}  {r['hero_model'] or 'hero':<34}  "
                   f"{fmt(r['iptm'])}  {fmt(None)}  {r['binder_sequence']}")
    txt += [
        "",
        "# [full_pipeline] 最终 selected (MPNN 重设计 → 重折叠 → Rosetta)",
        f"{'rank':>4}  {'stage':<8}  {'iptm':>8}  {'dG':>8}  "
        f"{'score':>8}  binder_sequence",
    ]
    for rank, r in enumerate(result["selected"], 1):
        txt.append(f"{rank:>4}  {r.stage:<8}  {fmt(r.refold_iptm or r.iptm)}  "
                   f"{fmt(r.rosetta_dG)}  {fmt(r.selection_score)}  "
                   f"{r.binder_sequence}")
    (task_dir / "design_summary.txt").write_text("\n".join(txt) + "\n")

    # selected PDB 已由 pipeline 落盘, 逐个注入 rosetta_dG REMARK (按 rank 对齐)
    sel_dir = full_dir / "selected_structures"
    for rank, r in enumerate(result["selected"], 1):
        if sel_dir.exists():
            for f in sorted(sel_dir.glob(f"{rank:04d}_*.pdb")):
                add_dg_remark(f, r.rosetta_dG)
    print(f"[e2e] design_summary.txt 与 PDB REMARK(dG) 已写入 {task_dir}")

    # ---- 全流程摘要 ----
    summary = {
        "task": args.task, "hotspot": args.hotspot, "seed": args.seed,
        "batch_size": args.batch_size,
        "hotspot_ids": hotspots,
        "target_chain_lengths": dict(zip(chain_order,
                                         [len(s) for s in target_sequence.split("|")])),
        "config": {"enable_mpnn": True, "enable_rosetta": True,
                   "mpnn_num_seqs": args.mpnn_num_seqs,
                   "hero_critic_names": cfg.hero_critic_names,
                   "use_stage1_scaling": False, "batch_size": args.batch_size,
                   "epitope_weight": args.epitope_weight if args.epitope_weight is not None else 0.5,
                   "dtype": "trunk bf16 (native) + heads fp32"},
        "elapsed_s": round(elapsed, 1),
        "n_stage1_candidates": len(s1_rows),
        "n_selected": len(result["selected"]),
        "n_quality_passed": len(result["quality_passed"]),
        "n_quality_failed": len(result["quality_failed"]),
        "n_refold_results": len(result["refold_results"]),
        "best_sequences": result["best_sequences"],
        "peak_vram_gib": max(
            (float(v.get("peak_allocated_gib", 0.0))
             for v in result["trajectory"].values()),
            default=0.0,
        ),
    }
    (task_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[e2e] DONE in {elapsed/60:.1f} min — {task_dir}")
    print(f"[e2e] stage1 候选 {len(s1_rows)}, selected {len(result['selected'])}, "
          f"quality {len(result['quality_passed'])}/{len(result['quality_passed'])+len(result['quality_failed'])}")


if __name__ == "__main__":
    main()
