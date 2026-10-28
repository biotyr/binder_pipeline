"""
ESMFold2 重折叠核心模块。

将 MPNN 逆折叠产生的 binder 序列与 target 序列配对,
通过 ESMFold2 重预测复合体结构, 输出 distogram + (可选) 全原子坐标。

本模块是 ``screen.py`` 的底层依赖, 也可独立使用进行单条序列的折叠验证。

依赖: esmcp/binder_design.py (ESMFold2 pipeline)
"""

from __future__ import annotations

from typing import Any

import torch

# ---- 从 esmcp 复用核心函数 ----
# 这些导入假设 esmcp/binder_design.py 在 PYTHONPATH 中。
# 如果直接运行, 确保 pipeline/ 目录可被导入:
#   import sys; sys.path.insert(0, "/home/taoyr/projects/pipeline")
from esmcp.binder_design import (
    PROTEIN_1TO3,
    TOKENS,
    build_complex,
    build_target_one_hot,
    compute_distogram_iptm_proxy,
    fold_and_get_distogram,
    merge_pair_chains_iptm,
    pdb_with_quality_remarks,
    sequence_to_one_hot,
)


def binder_seq_to_design(binder_seq: str) -> torch.Tensor:
    """将 binder 氨基酸序列转换为 ESMFold2 的 design 张量格式。

    输出可直接传入 ``fold_and_get_distogram`` 的 ``design`` 参数。

    Parameters
    ----------
    binder_seq : str
        单字母氨基酸序列, 如 "GFSLKYD...".

    Returns
    -------
    torch.Tensor
        shape (1, L, 20), dtype float32, 可微 one-hot。
    """
    return sequence_to_one_hot(binder_seq)[..., 2:22]


def refold_distogram(
    model: Any,
    target_seq: str,
    binder_seq: str,
    num_sampling_steps: int = 1,
    num_loops: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """用 ESMFold2 对 (target, binder) 复合体做一次前向折叠, 返回 distogram。

    等价于 ``fold_and_get_distogram`` 的薄封装, 额外返回
    parsed 后的 binder 序列信息。

    Parameters
    ----------
    model : ESMFold2ExperimentalModel
        已加载的 ESMFold2 模型。
    target_seq : str
        Target 氨基酸序列 (支持 "|" 分隔多链)。
    binder_seq : str
        Binder 氨基酸序列 (单链)。
    num_sampling_steps : int
        扩散采样步数: 1 = Scaling 快速模式, 200 = Hero 精确模式。
    num_loops : int
        回收迭代次数, 默认 1。
    calculate_confidence : bool
        True 时额外输出 pTM/iPTM/pLDDT (仅 Hero 模式有意义)。
    seed : int | None
        随机种子。

    Returns
    -------
    dict
        ``fold_and_get_distogram`` 的原始输出, 外加:
        - "binder_seq": str
        - "target_seq": str
        - "target_length": int
        - "binder_length": int
    """
    design = binder_seq_to_design(binder_seq)
    ##### claude 2026/8/13 start #####
    # Multi-chain targets: each "|"-separated chain is encoded independently.
    target_one_hot = build_target_one_hot(target_seq)
    ##### claude 2026/8/13 end #####

    fold_result = fold_and_get_distogram(
        model,
        target_seq,
        target_one_hot,
        design,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        calculate_confidence=calculate_confidence,
        seed=seed,
    )

    fold_result["binder_seq"] = binder_seq
    fold_result["target_seq"] = target_seq
    fold_result["target_length"] = len(target_seq.replace("|", ""))
    fold_result["binder_length"] = len(binder_seq)
    return fold_result


def refold_full(
    model: Any,
    target_seq: str,
    binder_seq: str,
    num_sampling_steps: int = 200,
    num_loops: int = 3,
    seed: int | None = None,
    is_antibody: bool = False,
) -> dict[str, Any]:
    """Hero 模式重折叠: 输出全原子 PDB + 全套置信度指标。

    这是完整版 refolding——200 步扩散 + 置信度计算 + PDB 构建 +
    REMARK 标注。结果可直接送入 ``bindcraft_filter`` 做物理评分。

    Parameters
    ----------
    model : ESMFold2ExperimentalModel
        已加载的 Hero Critic 模型。
    target_seq : str
        Target 序列。
    binder_seq : str
        Binder 序列。
    num_sampling_steps : int
        扩散步数, 默认 200 (Hero 精度)。
    num_loops : int
        回收迭代次数, 默认 3 (与 critic 评分一致)。
    seed : int | None
    is_antibody : bool
        binder 是否为抗体 (影响 CDR distogram 计算)。

    Returns
    -------
    dict
        包含:
        - "binder_seq": str
        - "target_seq": str
        - "complex": ProteinComplex | None
        - "annotated_pdb": str | None   (含 REMARK 质量标注)
        - "iptm": float | None
        - "ptm": float | None
        - "pair_chains_iptm": float | None   (binder-target t2b 残基加权合并值)
        - "pair_chains_iptm_detail": dict | None   (b2t/逐链, 仅诊断)
        - "ipsae": float | None   (ipSAE d0res 主分数, t2b; 补充信号)
        - "ipsae_detail": dict | None   (锚残基/b2t/逐链, 仅诊断)
        - "distogram_iptm_proxy": float
        - "cdr_distogram_iptm_proxy": float
        - "distogram_logits": Tensor
        - "plddt": float | None   (全局平均 pLDDT)
    """
    fold = refold_distogram(
        model,
        target_seq,
        binder_seq,
        num_sampling_steps=num_sampling_steps,
        num_loops=num_loops,
        calculate_confidence=True,
        seed=seed,
    )

    # Build ProteinComplex from coordinates
    pred_complex = build_complex(fold["inputs"], fold["output"])

    # Extract confidence metrics
    iptm_tensor = fold.get("iptm")
    iptm = iptm_tensor.item() if iptm_tensor is not None else None
    ptm_tensor = fold.get("ptm")
    ptm = ptm_tensor.item() if ptm_tensor is not None else None

    # Global pLDDT
    plddt_tensor = fold.get("plddt")
    ##### claude 2026/9/11 fix #####
    # 模型输出的 plddt 是逐 token 张量 ([L] 或 [B, L]), 不是标量 —
    # .item() 会抛 "a Tensor with N elements cannot be converted to
    # Scalar" (仅 Stage 4 refold_full 触发; Stage 1 score_critic 不提取
    # plddt 故被掩盖)。全局平均 pLDDT = 对所有 token 取均值。
    plddt = (
        plddt_tensor.mean().item()
        if plddt_tensor is not None
        else None
    )
    ##### claude 2026/9/11 fix end #####

    # Distogram-based binding confidence
    iptm_proxy = compute_distogram_iptm_proxy(
        fold["distogram_logits"],
        fold["target_length"],
        binder_seq,
        is_antibody,
    )

    ##### claude 2026/10/19 start #####
    # Binder-target pair ipTM (ConfidenceHead pair_chains_iptm), 与 Stage-1
    # score_critic 同源同式: 消费方向 [target][binder] (t2b), 多链 target 按
    # 残基加权合并; b2t/逐链仅诊断。
    pair_detail: dict | None = None
    pair_matrix = fold["output"].get("pair_chains_iptm")
    if pair_matrix is not None:
        pair_detail = merge_pair_chains_iptm(
            pair_matrix, [len(c) for c in target_seq.split("|")]
        )
    # ipSAE (补充信号): 同一 output["pae"] 直接计算 (ipsae.py 为 pipeline
    # 根目录共享度量模块, 根目录已在 sys.path — 本文件 esmcp import 同前提)。
    from ipsae import compute_ipsae

    ipsae_detail: dict | None = None
    pae_matrix = fold["output"].get("pae")
    if pae_matrix is not None:
        ipsae_detail = compute_ipsae(
            pae_matrix, [len(c) for c in target_seq.split("|")]
        )
    ##### claude 2026/10/19 end #####

    # Annotate PDB with quality remarks
    annotated_pdb: str | None = None
    if pred_complex is not None:
        annotated_pdb = pdb_with_quality_remarks(
            pred_complex.to_pdb_string(),
            iptm=iptm,
            ptm=ptm,
            distogram_binding_confidence=iptm_proxy.get(
                "distogram_iptm_proxy"
            ),
            ##### claude 2026/10/19 start #####
            pair_chains_iptm=(
                pair_detail["merged_t2b"] if pair_detail else None
            ),
            pair_chains_iptm_b2t=(
                pair_detail["merged_b2t"] if pair_detail else None
            ),
            pair_chains_iptm_chains=(
                ",".join(
                    f"{label}:{vals['t2b']:.4f}"
                    for label, vals in pair_detail["chains"].items()
                )
                if pair_detail
                else None
            ),
            ##### claude 2026/10/19 end #####
            ##### claude 2026/10/19 start #####
            ipsae=ipsae_detail["ipsae_t2b"] if ipsae_detail else None,
            ipsae_b2t=ipsae_detail["ipsae_b2t"] if ipsae_detail else None,
            ipsae_anchor=(
                ipsae_detail["anchor_t2b"] if ipsae_detail else None
            ),
            ##### claude 2026/10/19 end #####
        )

    return {
        "binder_seq": binder_seq,
        "target_seq": target_seq,
        "target_length": fold["target_length"],
        "binder_length": fold["binder_length"],
        "complex": pred_complex,
        "annotated_pdb": annotated_pdb,
        "iptm": iptm,
        "ptm": ptm,
        ##### claude 2026/10/19 start #####
        "pair_chains_iptm": (
            pair_detail["merged_t2b"] if pair_detail else None
        ),
        "pair_chains_iptm_detail": pair_detail,
        "ipsae": ipsae_detail["ipsae_t2b"] if ipsae_detail else None,
        "ipsae_detail": ipsae_detail,
        ##### claude 2026/10/19 end #####
        "plddt": plddt,
        "distogram_iptm_proxy": iptm_proxy.get("distogram_iptm_proxy"),
        "cdr_distogram_iptm_proxy": iptm_proxy.get("cdr_distogram_iptm_proxy"),
        "distogram_logits": fold["distogram_logits"],
    }
