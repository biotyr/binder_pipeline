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
# ]
# ///
"""
集成 Pipeline 核心模块 — 纯 Python, 零 Modal 依赖。

包含: PipelineConfig, PipelineResult, IntegratedPipeline,
      compute_selection_score, apply_filters, select_top_n

使用方式 (本地):
  from pipeline.core import IntegratedPipeline, PipelineConfig
  cfg = PipelineConfig(enable_mpnn=True)
  pipeline = IntegratedPipeline(cfg)
  result = pipeline.run(target_name="pd-l1", binder_name="minibinder")
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch



# ── 确保 pipeline/ 包可被导入 ──────────────────────────────────────
_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

# ── Pipeline 模块 (延迟导入以兼容 Modal 环境) ────────────────────────


def _ensure_esmcp_imports():
    """确保 esmcp/binder_design.py 的依赖在 Modal 镜像中已安装。"""
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput  # noqa: F401
    del torch, transformers, ProteinInput, StructurePredictionInput


# ── 数据结构 ────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """端到端 Pipeline 配置。

    Attributes
    ----------
    enable_mpnn : bool
        是否启用 ProteinMPNN 逆折叠优化。
    enable_rosetta : bool
        是否启用 Rosetta 物理评分 (需要 PyRosetta 许可)。
    mpnn_num_seqs : int
        MPNN 每个 PDB 生成的候选序列数。
    mpnn_sampling_temp : float
        MPNN 采样温度。
    mpnn_fix_interface : bool
        MPNN 是否固定界面残基。
    scaling_threshold : float
        Scaling 快筛的 distogram_iptm_proxy 最低阈值。
    quality_min_iptm_single : float
        Stage-2 pair 门下限 (binder-target pair_chains_iptm, 单链 target),
        默认 0.30 (统一粗筛线, 2026/10/19)。
    quality_min_iptm_multi : float
        Stage-2 pair 门下限 (多链 target), 默认 0.30 (与单链统一;
        精细鉴别由 min_ipsae=0.5 补充门承担)。
    quality_min_distogram_binding_confidence : float
        PDB 质量闸门的最低界面 distogram 置信度 (纯 binder×target
        界面, 无 target-target 稀释), 默认 0.50。
    quality_min_binder_plddt : float
        PDB 质量闸门的最低 binder pLDDT。
    iptm_metric : str
        Stage-6 排序的 ipTM 项指标源 ("ipsae" | "pair_chains" | "global"),
        见字段注释; 不控制 Stage-2 质量门。
    top_n : int
        最终输出的 Top-N 设计数。
    output_dir : str
        输出目录。
    save_pdbs : bool
        是否保存 PDB 文件。
    verbose : bool
    """

    # MPNN
    enable_mpnn: bool = True
    mpnn_num_seqs: int = 20
    mpnn_sampling_temp: float = 0.1
    mpnn_fix_interface: bool = True

    # Quality gate
    ##### claude 2026/10/19 start #####
    # pair 门阈值 (Stage-2 粗筛门, 与 iptm_metric 排序开关解耦; 2026-10-19
    # 用户定稿: 单链/多链统一 0.3)。角色为粗底线 —— pair_chains_iptm 的
    # d0 耦合全复合体长度 (跨 binder 长度/复杂度尺度不稳), 且精细鉴别
    # 职责已由 ipSAE 补充门 (0.5) 承担。回放参照 (diag_pair_iptm_replay,
    # 8 产物): 验证真解 pair 0.49-0.68 (过门), 退化 wrapper 0.256 /
    # seed4 0.198 / seed2 0.060 (拦下), founder 型缺陷设计 pair 0.494
    # 过本门而由 ipSAE 门拦截 —— 两门互补。块均值系统性低于旧全局
    # max-over-rows 值 (-0.2~-0.3), 勿与旧数值混比。
    quality_min_iptm_single: float = 0.30
    quality_min_iptm_multi: float = 0.30
    # ipSAE 补充门 (Stage-2, 与 iptm_metric 开关无关常透传; REMARK 键
    # "ipsae")。定标 (2026-10-19, diag_pair_iptm_replay, 8 产物 cutoff15):
    # 全流程验证真解 (3 个 selected) 0.797-0.813; 未验证/带缺陷产物
    # (founder: pair 值过门但 Stage-6 pLDDT FAIL + 序列单调; seed3 热点
    # 假阳性; wrapper 低复杂度) 0.15-0.38; 灾难失败 0.00-0.02 → 阈值 0.5
    # (分离带 (0.38, 0.80) 内偏宽松, 真解留 ~0.3 边际; n=3 待更多 e2e 复核)。
    # 与 pair 门互补: 拦截 founder 型 "pair 值过门但整体质量缺陷" 设计。
    min_ipsae: float | None = 0.5
    ##### claude 2026/10/19 end #####
    # 界面级质量闸门: distogram binding confidence (纯 binder×target
    # 界面熵, 无 target-target 稀释), 与 Stage 4 scaling 阈值惯例一致。
    quality_min_distogram_binding_confidence: float = 0.50
    ##### claude 2026/8/13 end #####
    quality_min_binder_plddt: float = 0.70

    ##### claude 2026/10/19 start #####
    # Stage-6 selection score 的 ipTM 项指标源 + Stage-4 hero 比较/排序源
    # (2026-10-19 定稿默认 "ipsae"; **仅控制排序, 不控制 Stage-2 门** ——
    # 质量门固定 pair (0.50/0.40) + min_ipsae (0.5) 双门, Average_i_pTM
    # 过滤槽固定 pair 值, 均与本开关解耦):
    # "ipsae" = ipSAE (d0res 主分数 t2b; 8 产物回放实证排序与验证状态
    #           一致: 验证真解 0.80+ > 未验证/带缺陷 0.15-0.38 > 灾难
    #           ≤0.02; 缺失回退 pair → 全局);
    # "pair_chains" = binder-target pair_chains_iptm (t2b 残基加权合并,
    #           组装置信度语义);
    # "global" = 全局 ipTM (诊断; 多链被 target-target 对稀释虚高)。
    iptm_metric: str = "ipsae"
    ##### claude 2026/10/19 end #####

    # Refold screening
    scaling_threshold: float = 0.50
    # Hero critics (Stage 4 重折叠精测使用的模型, 从 Stage 1 加载的 4 个中选择)
    hero_critic_names: list[str] = field(
        default_factory=lambda: ["ESMFold2-Experimental-Cutoff2025"]
    )

    # Rosetta
    enable_rosetta: bool = False
    rosetta_dssp_path: str = ""
    rosetta_relax_first: bool = True

    # Output
    top_n: int = 84
    output_dir: str = "./pipeline_output"
    save_pdbs: bool = True
    verbose: bool = True


@dataclass
class PipelineResult:
    """单条设计的完整评估结果。"""

    # Identity
    designed_sequence: str = ""  # "TARGET|BINDER"
    binder_sequence: str = ""
    target_name: str = ""
    binder_name: str = ""
    seed: int = 0
    stage: str = ""  # "esmfold2" | "mpnn"

    ##### claude 2026/8/13 start #####
    # 链元数据 (多链 target 支持): target_chains 为 "|" 拆分后的 target 链
    # 列表, chain_labels 为复合物全部链名 (含 binder, 按 asym_id 顺序),
    # binder_label 为 binder 链名 (多链 target 时不再固定 "B")。
    target_chains: list[str] = field(default_factory=list)
    chain_labels: list[str] = field(default_factory=list)
    binder_label: str = ""
    ##### claude 2026/8/13 end #####

    # ESMFold2 原生指标
    iptm: float | None = None
    ptm: float | None = None
    distogram_iptm_proxy: float | None = None
    final_loss: float | None = None

    ##### claude 2026/10/19 start #####
    # binder-target pair ipTM (t2b 方向残基加权合并值; iptm_metric=
    # "pair_chains" 时为 Stage-2/6 置信度槽位的指标源)。b2t/逐链明细
    # 不进 dataclass, 留在源结果 dict。
    pair_chains_iptm: float | None = None
    # ipSAE (d0res 主分数 t2b, 补充信号): Stage-2 补充门 + Stage-6
    # iptm_metric="ipsae" 排序源; anchor 为消费方向锚残基 flat 0-idx。
    ipsae: float | None = None
    ipsae_anchor: int | None = None
    ##### claude 2026/10/19 end #####

    # 重折叠后指标
    refold_iptm: float | None = None
    refold_ptm: float | None = None
    refold_plddt: float | None = None
    hero_model_name: str = ""

    # Rosetta 物理指标 (仅在 enable_rosetta=True 时有值)
    rosetta_dG: float | None = None
    rosetta_sc: float | None = None
    rosetta_hbonds: int | None = None
    rosetta_unsat_hbonds: int | None = None
    rosetta_clashes: int | None = None
    rosetta_binder_energy: float | None = None

    # 过滤状态
    filter_passed: bool = False
    filter_unmet: list[str] = field(default_factory=list)

    # 排序评分
    selection_score: float = 0.0
    isoelectric_point: float | None = None

    # PDB
    annotated_pdb: str | None = None
    complex: Any = None


# ── 排名 & 筛选 (复现 BindCraft 逻辑) ──────────────────────────────


def compute_selection_score(
    results: list[PipelineResult],
    rosetta_weight: float = 0.0,
    iptm_metric: str = "ipsae",
) -> list[PipelineResult]:
    """计算每条设计的综合选择评分。

    复现 ESMFold2 notebook 的排名逻辑:
      selection_score = 0.5 * iptm + 0.5 * distogram_iptm_proxy

    当 Rosetta 指标可用时 (rosetta_weight > 0), 引入物理评分:
      selection_score = (1 - rosetta_weight) * confidence_score
                      + rosetta_weight * physics_score

    其中:
      confidence_score = 0.5 * norm_iptm + 0.5 * norm_proxy
      physics_score    = 0.4 * (1 - norm_dG) + 0.3 * norm_sc
                        + 0.2 * norm_hbonds - 0.1 * norm_clashes

    Parameters
    ----------
    results : list[PipelineResult]
    rosetta_weight : float
        Rosetta 物理评分在综合评分中的权重 (0-1), 默认 0。
    iptm_metric : str
        selection score 的 ipTM 项指标源 (仅排序; Stage-2 门不随此项):
        "ipsae" 用 ipSAE (缺失回退 pair → 全局); "pair_chains" 用 pair
        值 (缺失回退全局); "global" 用 refold/全局 ipTM。

    Returns
    -------
    list[PipelineResult]
        原地更新 selection_score 并返回, 按评分降序。
    """
    if not results:
        return results

    # ── 置信度部分 ──
    ##### claude 2026/10/19 start #####
    if iptm_metric == "pair_chains":
        iptm_vals = np.array([
            (r.pair_chains_iptm or r.refold_iptm or r.iptm or 0.0)
            for r in results
        ])
    elif iptm_metric == "ipsae":
        # ipSAE 排序源; 旧行缺失回退 pair 值再回退全局 (过渡兼容)
        iptm_vals = np.array([
            (r.ipsae or r.pair_chains_iptm or r.refold_iptm or r.iptm or 0.0)
            for r in results
        ])
    else:
        iptm_vals = np.array([
            (r.refold_iptm or r.iptm or 0.0) for r in results
        ])
    ##### claude 2026/10/19 end #####
    proxy_vals = np.array([
        (r.distogram_iptm_proxy or 0.0) for r in results
    ])

    iptm_norm = _minmax_norm(iptm_vals)
    proxy_norm = _minmax_norm(proxy_vals)
    confidence_scores = 0.5 * iptm_norm + 0.5 * proxy_norm

    # ── Rosetta 物理部分 ──
    has_rosetta = any(r.rosetta_dG is not None for r in results)

    if has_rosetta and rosetta_weight > 0:
        dG_vals = np.array([
            r.rosetta_dG if r.rosetta_dG is not None else 0.0
            for r in results
        ])
        sc_vals = np.array([
            r.rosetta_sc if r.rosetta_sc is not None else 0.0
            for r in results
        ])
        hbond_vals = np.array([
            float(r.rosetta_hbonds or 0) for r in results
        ])
        clash_vals = np.array([
            float(r.rosetta_clashes or 0) for r in results
        ])

        # dG: 越低越好 → 反转
        dG_score = 1.0 - _minmax_norm(dG_vals)
        sc_score = _minmax_norm(sc_vals)
        hbond_score = _minmax_norm(hbond_vals)
        clash_penalty = _minmax_norm(clash_vals)

        physics_scores = (
            0.4 * dG_score
            + 0.3 * sc_score
            + 0.2 * hbond_score
            - 0.1 * clash_penalty
        )
        physics_scores = np.clip(physics_scores, 0.0, 1.0)

        for i, r in enumerate(results):
            r.selection_score = float(
                (1.0 - rosetta_weight) * confidence_scores[i]
                + rosetta_weight * physics_scores[i]
            )
    else:
        for i, r in enumerate(results):
            r.selection_score = float(confidence_scores[i])

    results.sort(key=lambda r: r.selection_score, reverse=True)
    return results


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max 归一化到 [0, 1]。常量数组返回全 0.5。"""
    if arr.max() - arr.min() < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min())


def apply_filters(
    results: list[PipelineResult],
    filters_json: str | None = None,
    pI_max: float = 6.0,
) -> tuple[list[PipelineResult], list[PipelineResult]]:
    """执行 BindCraft 风格的阈值过滤 + 等电点筛选。

    复现原 BindCraft 二元过滤逻辑:
    1. check_filters(data_dict, filters) → PASSED/FAILED
    2. 对 minibinder 做 pI < pI_max 筛选 (来自 notebook)

    Parameters
    ----------
    results : list[PipelineResult]
    filters_json : str | None
        filter 阈值 JSON 路径。None 使用内置 default_filters.json。
    pI_max : float
        minibinder 最高允许等电点, 默认 6.0。

    Returns
    -------
    passed : list[PipelineResult]
    failed : list[PipelineResult]
    """
    from bindercraft_filter.filter_check import check_filters

    if filters_json is None:
        filters_json = os.path.join(
            _pipeline_dir, "bindercraft_filter", "default_filters.json"
        )

    with open(filters_json, "r") as fh:
        filters = json.load(fh)

    passed: list[PipelineResult] = []
    failed: list[PipelineResult] = []

    for r in results:
        # 构建 data_dict
        data_dict = _build_data_dict(r)

        # BindCraft 阈值检查
        filter_result = check_filters(data_dict, filters)
        if filter_result is True:
            r.filter_passed = True
            r.filter_unmet = []
        else:
            r.filter_passed = False
            r.filter_unmet = filter_result

        # 等电点筛选 (复现 notebook)
        pI_ok = True
        if r.isoelectric_point is not None and r.isoelectric_point >= pI_max:
            if r.binder_name == "minibinder" or "minibinder" in r.binder_name:
                pI_ok = False
                if "pI" not in r.filter_unmet:
                    r.filter_unmet.append(f"pI={r.isoelectric_point:.1f}>={pI_max}")

        if r.filter_passed and pI_ok:
            passed.append(r)
        else:
            if not pI_ok:
                r.filter_passed = False
            failed.append(r)

    return passed, failed


def _build_data_dict(
    result: PipelineResult,
) -> dict[str, Any]:
    """从 PipelineResult 构建 check_filters 所需的 data_dict。

    仅使用有实际测量值的指标, 缺失的用 None (check_filters 会跳过)。
    Average_i_pTM 槽**固定 pair 值优先** (Stage-6 过滤阈值 0.4 按 pair 值
    定标, 不随 iptm_metric 排序开关切换; 旧行缺失回退 refold/全局)。
    """
    d: dict[str, Any] = {}

    # 置信度指标 (前缀 Average_, 模拟 BindCraft 的多模型平均)
    ##### claude 2026/10/19 start #####
    if result.pair_chains_iptm is not None:
        d["Average_i_pTM"] = result.pair_chains_iptm
    elif result.refold_iptm is not None:
        d["Average_i_pTM"] = result.refold_iptm
    elif result.iptm is not None:
        d["Average_i_pTM"] = result.iptm
    ##### claude 2026/10/19 end #####

    if result.refold_ptm is not None:
        d["Average_pTM"] = result.refold_ptm
    ##### claude 2026/9/16 start #####
    # 真实 pLDDT 闸: "Average_pLDDT" 槽位 (default_filters.json, ≥0.8) 改喂
    # refold_plddt (hero 重折叠全局平均 pLDDT), 不再借用 distogram_iptm_proxy
    # (旧实现名实不符, 实为 proxy≥0.8 闸)。stage1 (esmfold2) 条目无
    # refold_plddt → 该闸跳过; 其 pLDDT 已由 Stage-2 质量门 (≥0.70) 管控。
    if result.refold_plddt is not None:
        d["Average_pLDDT"] = result.refold_plddt
    ##### claude 2026/9/16 start #####
    # proxy 闸回归 (具名独立槽位): Average_pLDDT 改喂真 pLDDT 后, distogram
    # 折叠置信度代理改走自有标签 (default_filters.json ≥0.8, 与旧隐性闸一致)。
    if result.distogram_iptm_proxy is not None:
        d["Average_distogram_binding_confidence"] = result.distogram_iptm_proxy
    ##### claude 2026/9/16 end #####

    # Rosetta 指标
    if result.rosetta_dG is not None:
        d["Average_dG"] = result.rosetta_dG
    if result.rosetta_sc is not None:
        d["Average_ShapeComplementarity"] = result.rosetta_sc
    if result.rosetta_hbonds is not None:
        d["Average_n_InterfaceHbonds"] = result.rosetta_hbonds
    if result.rosetta_unsat_hbonds is not None:
        d["Average_n_InterfaceUnsatHbonds"] = result.rosetta_unsat_hbonds
    if result.rosetta_clashes is not None:
        d["Average_Relaxed_Clashes"] = result.rosetta_clashes
    if result.rosetta_binder_energy is not None:
        d["Average_Binder_Energy_Score"] = result.rosetta_binder_energy

    return d


def select_top_n(
    passed: list[PipelineResult],
    top_n: int = 84,
) -> list[PipelineResult]:
    """从通过过滤的设计中选取 Top-N。

    按 selection_score 降序, 每个 (target_name, binder_name) 组合
    最多取 top_n 条 (复现 notebook 的分组选择逻辑; stage 仅作为
    元数据列保留, 不参与分组)。
    """
    if not passed:
        return []

    df = pd.DataFrame([
        {
            "target_name": r.target_name,
            "binder_name": r.binder_name,
            "stage": r.stage,
            "designed_sequence": r.designed_sequence,
            "selection_score": r.selection_score,
            "result": r,
        }
        for r in passed
    ])

    # 按 (target, binder) 分组, 每组取 top_n
    selected: list[PipelineResult] = []
    for _group_key, group in df.groupby(["target_name", "binder_name"]):
        group_sorted = group.sort_values("selection_score", ascending=False)
        group_top = group_sorted.head(top_n)
        selected.extend(group_top["result"].tolist())

    selected.sort(key=lambda r: r.selection_score, reverse=True)
    return selected


# ── 主 Pipeline 类 ──────────────────────────────────────────────────


class IntegratedPipeline:
    """端到端 Binder 设计 Pipeline。

    在 Modal GPU 上依次执行:
    ESMFold2 设计 → MPNN 逆折叠 → 重折叠 → Rosetta 过滤 → 排序输出。
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    # ── 阶段 1: ESMFold2 设计 ─────────────────────────────────────

    def _run_esmfold2_design(
        self,
        target_name: str,
        binder_name: str,
        target_sequence: str | None,
        binder_sequence: str | None,
        is_antibody: bool | None,
        seed: int,
        batch_size: int,
        target_hotspot_ids: list[str] | None,
        epitope_contact_distance: float,
        inter_loss_mode: str | None = None,
        ##### claude 2026/10/21 start #####
        # None = auto: 热点任务默认启用 fold-rescue 三阶段调度
        # (2026-10-21 语义变更, 与 inter_loss_mode 同款 auto 约定);
        # 解析在 design_binder 内, 显式 False 可退出。
        fold_rescue_schedule: bool | None = None,
        ##### claude 2026/10/21 end #####
    ) -> tuple[list[str], Any, list[dict], Any]:
        """执行原始 ESMFold2 binder design。

        Returns
        -------
        best_sequences : list[str]
        trajectory : dict
        critic_results : list[dict]
        esm_design : ESMFold2Design
            已加载模型的 ESMFold2Design 实例, 供下游 Stage 复用。
            ``esm_design.hf_critic_models`` 可直接用于重折叠阶段。
        """
        from esmcp.binder_design import ESMFold2Design

        app_design = ESMFold2Design()
        ##### claude 2026/9/9 start #####
        # Stage-1 scaling critics (15 个 base300M/600M/6B checkpoint) 仅产出
        # is_scaling_critic=True 的诊断条目 (complex=None / 无 iPTM), 被
        # Stage 2/3 与 _collect_results 显式排除, 对 IntegratedPipeline 下游
        # 零影响; 且 base6B 组各自整载一个 6B 主干, 瞬态显存峰 ~+25GB/critic
        # (fp32 全程峰值 ~58GB, 仅 H100 可容)。默认关闭; 需 scaling 曲线
        # 诊断时在 H100 上改回 True, 或走 binder_design 自有入口显式传参。
        app_design.load(use_scaling_critics=False)
        ##### claude 2026/9/9 end #####

        best_sequences, trajectory, critic_results = app_design.design(
            target_name=target_name,
            binder_name=binder_name,
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            target_hotspot_ids=target_hotspot_ids,
            epitope_contact_distance=epitope_contact_distance,
            inter_loss_mode=inter_loss_mode,
            fold_rescue_schedule=fold_rescue_schedule,
        )

        # Stage 1 设计完成后, inversion models 和 ESMC LM head 已无用,
        # 释放以回收 ~30GB 显存供后续阶段使用。
        del app_design.inversion_models
        del app_design.esmc_model
        torch.cuda.empty_cache()

        return best_sequences, trajectory, critic_results, app_design

    # ── 阶段 2: 质量预过滤 ────────────────────────────────────────

    @staticmethod
    def _hotspot_contact_check(
        pdb_text: str,
        target_hotspot_ids: list[str],
        target_sequence: str,
        min_cb: float = 10.0,
        min_contacts: int = 3,
        contact_distance: float = 12.0,
    ) -> tuple[bool, str]:
        """热点几何核验 (2026-10-14 新增, 热点任务额外质量门)。

        现有质量门 (iPTM/pLDDT/proxy) 不度量热点 honoring——实证: 一个放弃
        热点 (28-30Å) 的 binder 可全线过门。本门要求**每个** hotspot 在
        hero 结构中被 binder 真实接触: 最近 binder Cβ (gly 用 CA) 距离
        ≤ ``min_cb`` 且 ``contact_distance`` 内 binder 残基数 ≥
        ``min_contacts``。

        输出 PDB 为复合体顺序编号 (靶链 A,B.. 在前, binder 最后一链), 热点
        的链内 1-indexed 序号 + 前序靶链长度 = flat 序号; 残基字母校验不符
        时在 ±2 内找回 (仅容差, 不改变判定对象)。
        """
        aa3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                  'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                  'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                  'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
        chain_lens = [len(c) for c in target_sequence.split("|")]
        atoms: dict[tuple[str, int], dict] = {}
        order: list[str] = []
        for ln in pdb_text.split("\n"):
            if not ln.startswith("ATOM"):
                continue
            el = ln[12:16].strip()
            if el not in ("CA", "CB"):
                continue
            ch, num = ln[21], int(ln[22:26])
            rec = atoms.setdefault(
                (ch, num),
                {"aa": aa3to1.get(ln[17:20].strip(), "?"), "cb": None,
                 "ca": None})
            if ch not in order:
                order.append(ch)
            rec["cb" if el == "CB" else "ca"] = (
                float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        if not order:
            return False, "PDB 无 ATOM 记录"
        binder = [v for (c, _), v in atoms.items() if c == order[-1]]
        reasons: list[str] = []
        for hs in target_hotspot_ids:
            ch, rid = hs.split(":")
            aa_letter, idx = rid[0], int(rid[1:])
            ci = order.index(ch) if ch in order else -1
            flat = sum(chain_lens[:ci]) + idx if ci >= 0 else idx
            rec = atoms.get((ch, flat))
            if rec is None or rec["aa"] != aa_letter:
                for k in (1, 2):  # ±2 容差找回
                    cand = atoms.get((ch, flat + k)) or atoms.get((ch, flat - k))
                    if cand is not None and cand["aa"] == aa_letter:
                        rec = cand
                        break
                    rec = None
            if rec is None or not (rec["cb"] or rec["ca"]):
                reasons.append(f"{hs}: 未定位")
                continue
            ref = rec["cb"] or rec["ca"]
            dists = sorted(math.dist(ref, b["cb"] or b["ca"]) for b in binder
                           if b["cb"] or b["ca"])
            n12 = sum(1 for d in dists if d < contact_distance)
            if not (dists and dists[0] <= min_cb and n12 >= min_contacts):
                reasons.append(
                    f"{hs}: min={dists[0]:.1f}Å n12={n12} "
                    f"(要求 ≤{min_cb}Å ∧ ≥{min_contacts})"
                )
        if reasons:
            return False, "; ".join(reasons)
        return True, "全部热点接触达标"

    def _quality_filter_critic_results(
        self,
        critic_results: list[dict],
        target_hotspot_ids: list[str] | None = None,
        target_sequence: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """对 Hero Critic 输出的 PDB 做质量预过滤。

        固定门组 (2026/10/19 定稿): pair_chains_iptm (统一 0.3 粗筛)
        + ipSAE 补充门 (min_ipsae=0.5) + binder pLDDT (0.70) +
        distogram proxy (0.50) + CA clash (=0); 全局 iPTM 仅记录不设门。
        ``target_hotspot_ids`` 非空时 (热点设计模式) 追加热点几何核验门
        (见 :meth:`_hotspot_contact_check`); 核验失败的条目进入 failed
        并记录原因。
        """
        from inverse_optimize.quality_filter import (
            QualityFilterConfig,
            filter_pdb_by_quality,
        )

        passed: list[dict] = []
        failed: list[dict] = []

        hero_results = [
            r for r in critic_results
            if not r.get("is_scaling_critic") and r.get("complex") is not None
        ]

        ##### claude 2026/8/13 start #####
        # 多链 target: binder 链名 = 复合物最后一条链的标签 (不再是固定的
        # "B")。所有 hero critic 共享同一 designed_sequence, 循环外推导一次。
        # pair 门阈值按 target 链数选择 (当前单链/多链统一 0.3, 字段保留
        # 以便日后分开调)。
        binder_label = ""
        target_chains: list[str] = []
        if hero_results:
            target_chains = [
                c for c in hero_results[0].get("designed_sequence", "").split("|")
                if c
            ][:-1]
            _, binder_label = _derive_chain_metadata(target_chains)
        min_iptm = (
            self.config.quality_min_iptm_multi
            if len(target_chains) > 1
            else self.config.quality_min_iptm_single
        )

        ##### claude 2026/10/19 start #####
        # Stage-2 置信度门固定双门 (2026-10-19 定稿): pair 门 (0.50/0.40,
        # REMARK 键 pair_chains_iptm) + ipSAE 补充门 (min_ipsae=0.5)。
        # 全局 iPTM 仅记录不设门; iptm_metric 只控制 Stage-6 排序源,
        # 与本门完全解耦。
        cfg = QualityFilterConfig(
            ##### claude 2026/10/19 end #####
            ##### claude 2026/10/19 start #####
            min_iptm=None,
            min_pair_chains_iptm=min_iptm,
            min_ipsae=self.config.min_ipsae,
            ##### claude 2026/10/19 end #####
            min_binder_plddt=self.config.quality_min_binder_plddt,
            ##### claude 2026/8/13 start #####
            # 界面级质量闸门 (distogram binding confidence, 来自 REMARK 250)
            min_distogram_binding_confidence=(
                self.config.quality_min_distogram_binding_confidence
            ),
            ##### claude 2026/8/13 end #####
            max_ca_clashes=0,
            binder_chain=binder_label,
        )
        ##### claude 2026/8/13 end #####

        for result in hero_results:
            # 使用 annotated_pdb (含 REMARK) 或导出 PDB
            pdb_str = result.get("annotated_pdb")
            if pdb_str is None:
                pdb_str = result["complex"].to_pdb_string()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", delete=False
            ) as tmp:
                tmp.write(pdb_str)
                pdb_path = tmp.name

            try:
                ##### claude 2026/8/13 start #####
                # binder_chain 与阈值已在 cfg 中按链数配置好
                qc = filter_pdb_by_quality(pdb_path, config=cfg)
                ##### claude 2026/8/13 end #####
                entry = {**result, "quality_check": qc}
                if qc["passed"] and target_hotspot_ids and target_sequence:
                    ##### claude 2026/10/14 start #####
                    # 热点设计模式的额外几何核验门 (质量门不度量热点
                    # honoring, 假阳性实证见 2026-10-14 多 seed 记录)。
                    hs_ok, hs_detail = self._hotspot_contact_check(
                        pdb_str, target_hotspot_ids, target_sequence
                    )
                    entry["hotspot_check"] = {"passed": hs_ok,
                                              "detail": hs_detail}
                    qc = {**qc, "passed": qc["passed"] and hs_ok}
                    ##### claude 2026/10/14 end #####
                if qc["passed"]:
                    passed.append(entry)
                else:
                    failed.append(entry)
            finally:
                os.unlink(pdb_path)

        if self.config.verbose:
            ##### claude 2026/8/13 start #####
            print(
                f"[Quality] {len(passed)}/{len(hero_results)} Hero Critic PDBs "
                f"passed quality gate "
                f"(pair_chains_iptm≥{min_iptm} "
                f"[{len(target_chains)} target chain(s)], "
                f"binder={binder_label}, "
                f"pLDDT≥{self.config.quality_min_binder_plddt}, "
                f"proxy≥{self.config.quality_min_distogram_binding_confidence}"
                + (
                    f", ipsae≥{self.config.min_ipsae}"
                    if self.config.min_ipsae is not None else ""
                ) + ")"
            )
            ##### claude 2026/8/13 end #####

        return passed, failed

    # ── 阶段 3: MPNN 逆折叠 ───────────────────────────────────────

    def _mpnn_redesign_passed(
        self,
        quality_passed: list[dict],
    ) -> list[dict]:
        """对通过质量检查的 PDB 执行 MPNN 重设计。"""
        from inverse_optimize.mpnn_redesign import mpnn_redesign_from_critic_results

        return mpnn_redesign_from_critic_results(
            quality_passed,
            num_seqs=self.config.mpnn_num_seqs,
            sampling_temp=self.config.mpnn_sampling_temp,
            fix_interface=self.config.mpnn_fix_interface,
        )

    # ── 阶段 4: 重折叠 & 筛选 ─────────────────────────────────────

    # 用于 Scaling 快筛的轻量 Critic — 仅做 1 步扩散 distogram,
    # 不加载 confidence head (esmc 复用 Stage 1 缓存的实例)。
    _SCALING_CRITIC_NAME: str = "ESMFold2-Experimental-Fast-base300M-step1500k"

    def _refold_mpnn_designs(
        self,
        mpnn_results: list[dict],
        target_seq: str,
        is_antibody: bool,
        hero_models: dict[str, Any],
    ) -> list[dict]:
        """对 MPNN 序列执行 Scaling → Hero 重折叠。

        - Scaling 阶段: 加载 **一个** 轻量 Scaling Critic (~300M base,
          1 步扩散, 仅 distogram), 用后即删。
        - Hero 阶段: 使用传入的 Hero Critics 字典 (200 步全原子 + 置信度),
          每条序列由所有 model 分别预测, 取 iPTM 最高者为最终结果。
          RefoldScreen 逐个 model 运行, 完成后释放该 model 的引用。

        与旧版不同, scaling 不再混用 4 个重型 Hero Critic。

        Parameters
        ----------
        mpnn_results : list[dict]
        target_seq : str
        is_antibody : bool
        hero_models : dict[str, Any]
            Stage 1 的 Hero Critic 模型字典 ``{name: ESMFold2ExperimentalModel}``。
            调用者应将所有权转移给此方法, 以便 RefoldScreen 逐模型释放显存。
        """
        from esmcp.binder_design import _load_hf_model
        from refold.screen import RefoldScreen, RefoldScreenConfig

        # 收集所有 MPNN 序列
        all_sequences: list[str] = []
        for mr in mpnn_results:
            for design in mr.get("mpnn_designs", []):
                all_sequences.append(design["sequence"])

        if not all_sequences:
            return []

        # ── 加载轻量 Scaling Critic ──
        if self.config.verbose:
            print(f"[Refold] Loading scaling critic: {self._SCALING_CRITIC_NAME}")

        scaling_critic = _load_hf_model(
            self._SCALING_CRITIC_NAME,
            lm_dropout=0.0,          # 1 步扩散无需 LM dropout
            cache_esmc=True,         # 复用 Stage 1 已缓存的 ESMC
            device="cuda",
        )

        screen_cfg = RefoldScreenConfig(
            scaling_threshold=self.config.scaling_threshold,
            is_antibody=is_antibody,
            ##### claude 2026/10/19 start #####
            iptm_metric=self.config.iptm_metric,
            ##### claude 2026/10/19 end #####
            verbose=self.config.verbose,
        )

        # 按配置补齐缺失的 Hero Critic (未在 Stage 1 加载的名称)
        for name in self.config.hero_critic_names:
            if name not in hero_models:
                if self.config.verbose:
                    print(f"[Refold] Loading hero critic on demand: {name}")
                hero_models[name] = _load_hf_model(
                    name, lm_dropout=0.25, cache_esmc=True, device="cuda",
                )

        try:
            screen = RefoldScreen(
                target_seq=target_seq,
                scaling_models={self._SCALING_CRITIC_NAME: scaling_critic},
                hero_models=hero_models,
                config=screen_cfg,
            )

            hero_results, scaling_failed = screen.run(all_sequences)
        finally:
            # Scaling 完成后立即释放 (~1-2GB), 为 Hero 阶段腾显存
            del scaling_critic
            torch.cuda.empty_cache()

        if self.config.verbose:
            print(
                f"[Refold] {len(hero_results)} hero predictions, "
                f"{len(scaling_failed)} failed scaling"
            )

        return hero_results

    # ── 阶段 5: Rosetta 物理评分 (可选) ───────────────────────────

    def _rosetta_score(
        self,
        hero_results: list[dict],
    ) -> list[dict]:
        """对 Hero 预测结果执行 Rosetta 物理评分。

        2026/9/16 起同时覆盖 MPNN 重折叠条目与 stage1 (critic) 设计条目:
        两者均为 "annotated_pdb + 末段为 binder" 的复合物结构。
        """
        if not self.config.enable_rosetta:
            return hero_results

        try:
            from bindercraft_filter.integration import screen_single_pdb
        except ImportError:
            print("[Rosetta] PyRosetta not available — skipping physical scoring.")
            return hero_results

        scored: list[dict] = []
        for result in hero_results:
            pdb_str = result.get("annotated_pdb")
            if pdb_str is None:
                continue

            ##### claude 2026/8/13 start #####
            # 链名推导: refold_results 条目携带 "target_seq" ("T1|T2" 形式),
            # binder 为复合物最后一条链 (多链 target 时不再固定 "B")。
            # critic (stage1) 条目无 "target_seq" 键, 从 "designed_sequence"
            # ("T1|T2|binder") 回退 — target 为除末段外全部链。(2026/9/16)
            target_seq = result.get("target_seq") or "|".join(
                result.get("designed_sequence", "").split("|")[:-1]
            )
            target_chains = [c for c in target_seq.split("|") if c] if target_seq else []
            if not target_chains:
                if self.config.verbose:
                    print(f"[Rosetta] Skip: no target_seq in result")
                continue
            ##### claude 2026/9/11 fix #####
            # target_chains 的元素是"链序列"而非链名 — split("|") 每段是
            # 整条氨基酸序列, 直接传给 screen_single_pdb 会在 hotspot_residues
            # 的 structure[0][chain] 处 KeyError(整段序列)。改用链数推导
            # asym_id 链名 (["A", ...]), binder 恒为末链。
            labels, binder_label = _derive_chain_metadata(target_chains)
            rosetta_target_labels = labels[:-1]
            ##### claude 2026/9/11 fix end #####
            # refold 条目用 "sequence" 键存 binder 序列 (非 "binder_sequence")
            binder_seq = result.get(
                "binder_sequence", result.get("sequence", "")
            )
            ##### claude 2026/9/16 start #####
            # critic (stage1) 条目两者皆无, binder 取 designed_sequence 末段
            if not binder_seq:
                segments = result.get("designed_sequence", "").split("|")
                binder_seq = segments[-1] if segments else ""
            ##### claude 2026/9/16 end #####
            ##### claude 2026/8/13 end #####

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", delete=False
            ) as tmp:
                tmp.write(pdb_str)
                pdb_path = tmp.name

            try:
                ##### claude 2026/8/13 start #####
                screening = screen_single_pdb(
                    pdb_path=pdb_path,
                    binder_sequence=binder_seq,
                    dssp_path=self.config.rosetta_dssp_path,
                    binder_chain=binder_label,
                    target_chains=rosetta_target_labels,
                    relax_first=self.config.rosetta_relax_first,
                )
                ##### claude 2026/8/13 end #####
                result["rosetta_screening"] = screening
            except Exception as exc:
                if self.config.verbose:
                    print(f"[Rosetta] Error: {exc}")
                result["rosetta_screening"] = None
            finally:
                os.unlink(pdb_path)

            scored.append(result)

        return scored

    # ── 主入口 ────────────────────────────────────────────────────

    def run(
        self,
        target_name: str,
        binder_name: str,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        target_hotspot_ids: list[str] | None = None,
        epitope_contact_distance: float = 12.0,
        inter_loss_mode: str | None = None,
        ##### claude 2026/10/21 start #####
        # None = auto: 热点任务默认启用 fold-rescue 三阶段调度
        # (2026-10-21 语义变更); 显式 False 可退出, 无热点时 True 报错。
        fold_rescue_schedule: bool | None = None,
        ##### claude 2026/10/21 end #####
    ) -> dict[str, Any]:
        """执行完整的集成 Pipeline。

        Returns
        -------
        dict
            - "best_sequences": list[str]  (ESMFold2 原始最佳序列)
            - "trajectory": dict  (ESMFold2 设计 trajectory)
            - "critic_results": list[dict]  (ESMFold2 原始 critic_results)
            - "quality_passed": list[dict]
            - "quality_failed": list[dict]
            - "mpnn_results": list[dict]
            - "refold_results": list[dict]
            - "final_results": list[PipelineResult]
            - "selected": list[PipelineResult]  (Top-N)
            - "elapsed_s": float
        """
        t_start = time.time()
        cfg = self.config

        ##### claude 2026/10/19 start #####
        assert cfg.iptm_metric in ("global", "pair_chains", "ipsae"), (
            f"iptm_metric must be 'global', 'pair_chains' or 'ipsae', "
            f"got {cfg.iptm_metric!r}"
        )
        ##### claude 2026/10/19 end #####

        # ── 阶段 1: ESMFold2 设计 ─────────────────────────────────
        if cfg.verbose:
            print(f"{'='*60}")
            print(f"[Pipeline] Stage 1: ESMFold2 Design")
            print(f"  target={target_name}  binder={binder_name}  seed={seed}")
            print(f"{'='*60}")

        best_sequences, trajectory, critic_results, esm_design = (
            self._run_esmfold2_design(
                target_name, binder_name,
                target_sequence, binder_sequence,
                is_antibody, seed, batch_size,
                target_hotspot_ids, epitope_contact_distance,
                inter_loss_mode, fold_rescue_schedule,
            )
        )
        # Stage 1 之后保留模型 (供 Stage 4 复用): 按 hero_critic_names 过滤,
        # 未启用的模型立即释放以回收显存。
        _names = cfg.hero_critic_names or ["ESMFold2-Experimental-Cutoff2025"]
        _hero_models = {
            name: esm_design.hf_critic_models[name]
            for name in _names
            if name in esm_design.hf_critic_models
        }
        _missing_hero = [n for n in _names if n not in esm_design.hf_critic_models]
        if _missing_hero and cfg.verbose:
            print(f"[Pipeline] Hero critics not loaded in Stage 1 "
                  f"(loaded on demand in Stage 4): {_missing_hero}")
        # 所有权转移: 清空 Stage 1 的 hf_critic_models 字典, 使 _hero_models
        # 成为唯一持有者, 这样 RefoldScreen 可以在每个模型完成后真正释放显存。
        esm_design.hf_critic_models.clear()
        torch.cuda.empty_cache()

        target_seq = (
            target_sequence
            if target_sequence
            ##### claude 2026/8/13 start #####
            # 多链 target: 保留除 binder 外的全部链 (binder 恒为最后一段)
            else "|".join(best_sequences[0].split("|")[:-1]) if best_sequences else ""
            ##### claude 2026/8/13 end #####
        )
        _binder_seq_original = (
            best_sequences[0].split("|")[-1] if best_sequences else ""
        )
        del _binder_seq_original  # reserved for future use

        # ── 阶段 2: 质量预过滤 ────────────────────────────────────
        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"[Pipeline] Stage 2: Quality Pre-filter")
            print(f"{'='*60}")

        quality_passed, quality_failed = self._quality_filter_critic_results(
            critic_results,
            target_hotspot_ids=target_hotspot_ids,
            target_sequence=target_sequence,
        )

        # ── 阶段 3: MPNN 逆折叠 ───────────────────────────────────
        mpnn_results: list[dict] = []
        if cfg.enable_mpnn and quality_passed:
            if cfg.verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Stage 3: MPNN Inverse Folding")
                print(f"{'='*60}")

            mpnn_results = self._mpnn_redesign_passed(quality_passed)

        # ── 阶段 4: 重折叠 ────────────────────────────────────────
        refold_results: list[dict] = []
        if mpnn_results and target_seq:
            if cfg.verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Stage 4: Refold & Screen")
                print(f"{'='*60}")

            refold_results = self._refold_mpnn_designs(
                mpnn_results, target_seq, bool(is_antibody),
                hero_models=_hero_models,
            )

            # 重折叠完成后释放 ESMFold2 模型 (~20GB), 为 Rosetta 腾空间
            del esm_design
            del _hero_models
            torch.cuda.empty_cache()

        # ── 阶段 5: Rosetta 物理评分 ──────────────────────────────
        ##### claude 2026/9/16 start #####
        # stage1 (critic) 设计与 MPNN 重折叠结果一并进入 Rosetta 队列,
        # Stage 6 排序/筛选对两类候选统一使用物理能量。
        if cfg.enable_rosetta and (refold_results or critic_results):
            if cfg.verbose:
                print(f"\n{'='*60}")
                print(f"[Pipeline] Stage 5: Rosetta Physical Scoring")
                print(f"  critic(stage1)={len(critic_results)}  "
                      f"refold(mpnn)={len(refold_results)}")
                print(f"{'='*60}")

            if critic_results:
                critic_results = self._rosetta_score(critic_results)
            if refold_results:
                refold_results = self._rosetta_score(refold_results)
        ##### claude 2026/9/16 end #####

        # ── 汇总为 PipelineResult ─────────────────────────────────
        all_results = _collect_results(
            critic_results=critic_results,
            quality_passed=quality_passed,
            refold_results=refold_results,
            target_name=target_name,
            binder_name=binder_name,
            seed=seed,
            is_antibody=bool(is_antibody),
        )

        # ── 计算排名评分 ──────────────────────────────────────────
        rosetta_w = 0.3 if cfg.enable_rosetta else 0.0
        all_results = compute_selection_score(
            all_results, rosetta_weight=rosetta_w, iptm_metric=cfg.iptm_metric
        )

        # ── 过滤 ──────────────────────────────────────────────────
        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"[Pipeline] Stage 6: Filter & Select")
            print(f"{'='*60}")

        passed, failed = apply_filters(all_results)

        # ── 选择 Top-N ───────────────────────────────────────────
        selected = select_top_n(passed, top_n=cfg.top_n)

        if cfg.verbose:
            print(f"  Filter: {len(passed)} passed / {len(failed)} failed")
            print(f"  Selected: Top-{cfg.top_n} = {len(selected)} designs")
            if selected:
                print(f"  Best score: {selected[0].selection_score:.4f}")
                print(f"  Best seq:   {selected[0].binder_sequence[:30]}...")

        # ── 保存输出 ──────────────────────────────────────────────
        if cfg.save_pdbs and selected:
            _save_outputs(selected, cfg)

        elapsed = time.time() - t_start
        if cfg.verbose:
            print(f"\n[Pipeline] Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

        return {
            "best_sequences": best_sequences,
            "trajectory": trajectory,
            "critic_results": critic_results,
            "quality_passed": quality_passed,
            "quality_failed": quality_failed,
            "mpnn_results": mpnn_results,
            "refold_results": refold_results,
            "final_results": all_results,
            "selected": selected,
            "elapsed_s": elapsed,
        }


# ── 结果汇总 ────────────────────────────────────────────────────────


##### claude 2026/8/13 start #####
def _derive_chain_metadata(target_chains: list[str]) -> tuple[list[str], str]:
    """按 asym_id 顺序推导复合物链名列表与 binder 链名。

    target 链占前部 (asym_id 0..n-1 → "A", "B", ...), binder 链
    紧随其后 (asym_id n)。多链 target 时 binder 链名不再固定 "B"。
    """
    from esmcp.binder_design import _asym_id_to_chain_label

    labels = [_asym_id_to_chain_label(i) for i in range(len(target_chains) + 1)]
    binder_label = labels[-1] if labels else ""
    return labels, binder_label
##### claude 2026/8/13 end #####


def _collect_results(
    critic_results: list[dict],
    quality_passed: list[dict],  # noqa: ARG001 reserved for future filtering
    refold_results: list[dict],
    target_name: str,
    binder_name: str,
    seed: int,
    is_antibody: bool,  # noqa: ARG001 reserved for future filtering
) -> list[PipelineResult]:
    """将各阶段结果汇总为统一 PipelineResult 列表。"""
    results: list[PipelineResult] = []

    # (A) 原始 ESMFold2 Hero Critic 结果 (未经 MPNN)
    hero_critics = [
        r for r in critic_results
        if not r.get("is_scaling_critic") and r.get("complex") is not None
    ]
    for r in hero_critics:
        chains = r.get("designed_sequence", "").split("|")
        binder_seq = chains[-1]
        ##### claude 2026/8/13 start #####
        target_chains = chains[:-1]
        labels, binder_label = _derive_chain_metadata(target_chains)
        ##### claude 2026/8/13 end #####
        results.append(PipelineResult(
            designed_sequence=r.get("designed_sequence", ""),
            binder_sequence=binder_seq,
            target_name=target_name,
            binder_name=binder_name,
            seed=seed,
            stage="esmfold2",
            hero_model_name=r.get("critic_name", ""),
            ##### claude 2026/8/13 start #####
            target_chains=target_chains,
            chain_labels=labels,
            binder_label=binder_label,
            ##### claude 2026/8/13 end #####
            iptm=r.get("iptm"),
            ptm=r.get("ptm"),
            distogram_iptm_proxy=r.get("distogram_iptm_proxy"),
            final_loss=r.get("final_loss"),
            ##### claude 2026/10/19 start #####
            pair_chains_iptm=r.get("pair_chains_iptm"),
            ipsae=r.get("ipsae"),
            ipsae_anchor=(
                (r.get("ipsae_detail") or {}).get("anchor_t2b")
                if r.get("ipsae_detail")
                else None
            ),
            ##### claude 2026/10/19 end #####
            annotated_pdb=r.get("annotated_pdb"),
            complex=r.get("complex"),
            ##### claude 2026/9/16 start #####
            # Rosetta: stage1 设计经 Stage 5 评分后携带 rosetta_screening,
            # 与 (B) 分支同式映射, 使 Stage 6 排序/筛选覆盖 stage1 候选。
            rosetta_dG=_safe_get_rosetta(r, "interface_dG"),
            rosetta_sc=_safe_get_rosetta(r, "interface_sc"),
            rosetta_hbonds=_safe_get_rosetta(r, "interface_interface_hbonds"),
            rosetta_unsat_hbonds=_safe_get_rosetta(r, "interface_delta_unsat_hbonds"),
            rosetta_clashes=_safe_get_rosetta(r, "Relaxed_Clashes"),
            rosetta_binder_energy=_safe_get_rosetta(r, "binder_score"),
            ##### claude 2026/9/16 end #####
            isoelectric_point=_compute_pI(binder_seq) if binder_seq else None,
        ))

    # (B) MPNN → 重折叠 结果
    for mr in refold_results:
        binder_seq = mr.get("binder_sequence", mr.get("sequence", ""))
        ##### claude 2026/8/13 start #####
        target_seq = mr.get("target_seq", "")
        target_chains = target_seq.split("|") if target_seq else []
        labels, binder_label = _derive_chain_metadata(target_chains)
        ##### claude 2026/8/13 end #####
        results.append(PipelineResult(
            designed_sequence=f"{mr.get('target_seq', '')}|{binder_seq}",
            binder_sequence=binder_seq,
            target_name=target_name,
            binder_name=binder_name,
            seed=seed,
            stage="mpnn",
            hero_model_name=mr.get("hero_model", ""),
            ##### claude 2026/8/13 start #####
            target_chains=target_chains,
            chain_labels=labels,
            binder_label=binder_label,
            ##### claude 2026/8/13 end #####
            refold_iptm=mr.get("iptm"),
            refold_ptm=mr.get("ptm"),
            refold_plddt=mr.get("plddt"),
            distogram_iptm_proxy=mr.get("distogram_iptm_proxy"),
            ##### claude 2026/10/19 start #####
            pair_chains_iptm=mr.get("pair_chains_iptm"),
            ipsae=mr.get("ipsae"),
            ipsae_anchor=(
                (mr.get("ipsae_detail") or {}).get("anchor_t2b")
                if mr.get("ipsae_detail")
                else None
            ),
            ##### claude 2026/10/19 end #####
            annotated_pdb=mr.get("annotated_pdb"),
            complex=mr.get("complex"),
            # Rosetta
            rosetta_dG=_safe_get_rosetta(mr, "interface_dG"),
            rosetta_sc=_safe_get_rosetta(mr, "interface_sc"),
            rosetta_hbonds=_safe_get_rosetta(mr, "interface_interface_hbonds"),
            rosetta_unsat_hbonds=_safe_get_rosetta(mr, "interface_delta_unsat_hbonds"),
            rosetta_clashes=_safe_get_rosetta(mr, "Relaxed_Clashes"),
            rosetta_binder_energy=_safe_get_rosetta(mr, "binder_score"),
            isoelectric_point=_compute_pI(binder_seq) if binder_seq else None,
        ))

    return results


def _compute_pI(sequence: str) -> float | None:
    """计算蛋白质等电点。"""
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return ProteinAnalysis(sequence).isoelectric_point()
    except Exception:
        return None


def _safe_get_rosetta(result: dict, key: str) -> Any:
    """安全地从 Rosetta screening 结果中提取指标。"""
    screening = result.get("rosetta_screening", {})
    if screening is None:
        return None
    iface = screening.get("interface_scores", {})
    if isinstance(iface, dict) and key in iface:
        return iface[key]
    return screening.get(key)


# ── 输出 ────────────────────────────────────────────────────────────


def _save_outputs(
    selected: list[PipelineResult],
    config: PipelineConfig,
) -> None:
    """保存选中的设计: PDB 文件 + 排名表格。"""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = out_dir / "selected_structures"
    pdb_dir.mkdir(exist_ok=True)

    # 排名表格
    rows = []
    for rank, r in enumerate(selected, start=1):
        rows.append({
            "rank": rank,
            "target_name": r.target_name,
            "binder_name": r.binder_name,
            "stage": r.stage,
            "hero_model": r.hero_model_name,
            ##### claude 2026/8/13 start #####
            # 链元数据: 多链 target 的可追溯性 (binder 链名 + target 链列表)
            "binder_label": r.binder_label,
            "target_chains": "|".join(r.target_chains),
            "chain_labels": ",".join(r.chain_labels),
            ##### claude 2026/8/13 end #####
            "binder_sequence": r.binder_sequence,
            "selection_score": round(r.selection_score, 4),
            "iptm": r.refold_iptm or r.iptm,
            "pair_chains_iptm": r.pair_chains_iptm,
            "ipsae": r.ipsae,
            "distogram_iptm_proxy": r.distogram_iptm_proxy,
            "rosetta_dG": r.rosetta_dG,
            "rosetta_sc": r.rosetta_sc,
            "filter_passed": r.filter_passed,
            "isoelectric_point": (
                round(r.isoelectric_point, 1) if r.isoelectric_point else None
            ),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "selection.parquet", index=False)
    print(f"[Output] Ranking table saved: {out_dir / 'selection.parquet'}")

    # PDB 文件
    for rank, r in enumerate(selected, start=1):
        pdb_str = r.annotated_pdb
        if pdb_str is None and r.complex is not None:
            try:
                pdb_str = r.complex.to_pdb_string()
            except Exception:
                continue
        if pdb_str is None:
            continue

        fname = (
            f"{rank:04d}_{r.target_name}_{r.binder_name}_"
            f"score{r.selection_score:.3f}.pdb"
        )
        (pdb_dir / fname).write_text(pdb_str)

    print(f"[Output] {len(selected)} PDBs saved: {pdb_dir.resolve()}")


