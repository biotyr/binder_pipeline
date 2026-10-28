"""
ipSAE — inter-chain PAE 界面分数 (Dunbrack lab ipsae.py v4 语义的自包含复刻)。

本 pipeline 使用的唯一变体: **设 cutoff 的主分数 ipSAE (d0res)** —— 逐锚残基
以 "PAE<cutoff 的伙伴数" 定 d0, 锚残基行均值取 max。方向约定全 pipeline 统一
(2026-10-19): **t2b = 行为 target 残基 (对齐参考)、列为 binder 残基** —— 固定
target 评估 binder 位置置信度; b2t 仅诊断记录, 不进任何消费路径。

正确性由 ``test/module_tests/test_ipsae_equivalence.py`` 钉死: 对参考实现
(/home/taoyr/projects/IPSAE-main/ipsae.py) 的 Example 数据 (cutoff 15/15)
做逐链对数值等价断言 (相对误差 < 1e-6)。修改本文件或参考脚本时必须重跑。

语义 (protein-only):
  ptm(PAE, d0) = 1/(1+(PAE/d0)²),  d0(L) = max(1, 1.24·(L−15)^⅓ − 1.8)
  锚残基 i (参考组): n0res(i) = #{j∈查询组: PAE[i,j] < cutoff}
                     d0res(i)  = d0(max(26, n0res(i)))
                     score(i)  = mean over valid j of ptm(PAE[i,j], d0res(i))
                                 (无 valid 对的锚 = 0)
  asym = max over 参考组锚残基 (argmax 与参考实现一致: 非参考行为 0 后取 argmax)

⚠️ 与 ESMFold2 ConfidenceHead 家族 (tm_expected / pair_chains_iptm) **数值
不同族, 不可互换**: 后者对 soft PAE bins 做期望且 d0 用全复合体长度; 本模块
用点值 PAE 与配对专属 d0。两者是同级补充信号, 各走各的定标阈值。

定位: 所有 hero 精确折叠处共用的共享度量模块 (Stage-1 score_critic 与
Stage-4 refold_full), 故置于 pipeline 根目录与 core.py 同级, 不属于任何
Stage 子包。纯 numpy, 零 torch / 零 pipeline 内部依赖。
"""

from __future__ import annotations

import string

import numpy as np

# AF2 惯例 cutoff (ipsae.py README 示例 15/15)。provisional — 回放定标后复核。
IPSAE_PAE_CUTOFF = 15.0

# 与 esmcp/binder_design._asym_id_to_chain_label 同表 (此处不 import, 保持零依赖)
_CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _chain_label(asym_id: int) -> str:
    """asym_id → 链名 (A, B, ..., Z, a, ...; 与 pipeline 链名约定一致)。"""
    label = ""
    n = len(_CHAIN_ALPHABET)
    while True:
        label = _CHAIN_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def _ptm(x: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """TM 形式变换 1/(1+(x/d0)²)。d0 与 x 逐元素对应 (逐锚 d0 广播)。"""
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def _d0_from_counts(n0res: np.ndarray) -> np.ndarray:
    """calc_d0_array (ipsae.py v4): L clamp ≥26, 下限 1.0。"""
    L = np.maximum(26, np.asarray(n0res, dtype=float))
    return np.maximum(1.0, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_d0res_pair(
    pae: np.ndarray,
    ref_mask: np.ndarray,
    qry_mask: np.ndarray,
    cutoff: float = IPSAE_PAE_CUTOFF,
) -> tuple[float, int, int, np.ndarray]:
    """单方向 ipSAE (d0res): ref 行 (锚) → qry 列。

    与参考实现逐行对齐:
      valid[i]  = qry_mask ∧ (PAE[i,:] < cutoff)   (严格小于)
      n0res(i)  = valid[i].sum()
      score(i)  = mean_{j∈valid[i]} ptm(PAE[i,j], d0res(i)); 无 valid → 0
      非参考行 score = 0;  asym = argmax(全数组) (参考实现同式)

    Returns
    -------
    (score, anchor_flat_idx, n0res_at_anchor, per_anchor_scores)
    """
    pae = np.asarray(pae, dtype=float)
    ref_mask = np.asarray(ref_mask, dtype=bool)
    qry_mask = np.asarray(qry_mask, dtype=bool)
    valid = qry_mask[None, :] & (pae < cutoff)          # [L, L]
    counts = valid.sum(axis=1)                           # [L]
    d0 = _d0_from_counts(counts)                         # [L]
    ptm_all = _ptm(pae, d0[:, None])                     # 逐锚 d0 广播
    sums = np.where(valid, ptm_all, 0.0).sum(axis=1)
    scores = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    scores = np.where(ref_mask, scores, 0.0)             # 非参考行 = 0 (同参考实现)
    anchor = int(np.argmax(scores))
    return float(scores[anchor]), anchor, int(counts[anchor]), scores


def compute_ipsae(
    pae,
    target_chain_lengths: list[int],
    cutoff: float = IPSAE_PAE_CUTOFF,
) -> dict:
    """binder-target 对的 ipSAE (d0res), t2b 主分数 + b2t/逐链诊断。

    Parameters
    ----------
    pae
        期望 PAE 矩阵 ``[L, L]`` 或 ``[B, L, L]`` (np.ndarray / torch tensor
        均可, 后者内部 ``detach().cpu().numpy()``)。token 序 = target 链
        (``split("|")`` 序) 在前、binder 在后 — 与 ``fold_and_get_distogram``
        的输出布局一致, 纯蛋白免 token 对齐。
    target_chain_lengths
        各 target 链残基数 (链内序), binder 占据末尾 ``L - sum`` 个 token。

    Returns
    -------
    dict
        ``{"ipsae_t2b": float,        # 消费值: 全 target 组 → binder
           "ipsae_b2t": float,        # 诊断: 反方向
           "anchor_t2b": int,         # 消费方向锚残基 flat 0-idx (结合链长
                                      # 可换算回链内序号, 用于界面定位)
           "n0res_at_anchor": int,
           "chains": {label: {"t2b", "b2t"}}}   # 单链对逐项, 诊断
        """
    if hasattr(pae, "detach"):       # torch tensor (避免 import torch)
        pae = pae.detach().cpu().numpy()
    pae = np.asarray(pae, dtype=float)
    if pae.ndim == 3:
        pae = pae[0]
    L = pae.shape[0]
    assert pae.shape[0] == pae.shape[1], f"PAE 非方阵: {pae.shape}"
    n_target = int(sum(target_chain_lengths))
    assert 0 < n_target < L, (
        f"target 残基数 {n_target} 必须小于总 token 数 {L} (binder 占尾部)。"
    )

    idx = np.arange(L)
    tgt_mask = idx < n_target
    bnd_mask = ~tgt_mask

    t2b, anchor, n0_anchor, _ = ipsae_d0res_pair(pae, tgt_mask, bnd_mask, cutoff)
    b2t, _, _, _ = ipsae_d0res_pair(pae, bnd_mask, tgt_mask, cutoff)

    chains: dict[str, dict[str, float]] = {}
    start = 0
    for ci, length in enumerate(target_chain_lengths):
        mask = (idx >= start) & (idx < start + int(length))
        s_t2b, _, _, _ = ipsae_d0res_pair(pae, mask, bnd_mask, cutoff)
        s_b2t, _, _, _ = ipsae_d0res_pair(pae, bnd_mask, mask, cutoff)
        chains[_chain_label(ci)] = {"t2b": s_t2b, "b2t": s_b2t}
        start += int(length)

    return {
        "ipsae_t2b": t2b,
        "ipsae_b2t": b2t,
        "anchor_t2b": anchor,
        "n0res_at_anchor": n0_anchor,
        "chains": chains,
    }
