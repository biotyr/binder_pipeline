"""
基于 PDB REMARK 结构质量指标的预过滤模块。

读取 ESMFold2 Critic 输出的 PDB, 从 REMARK 250 记录提取 iPTM/pTM,
从 B-factor 列提取逐残基 pLDDT, 结合 CA clash 检测,
在进入 ProteinMPNN 重设计前剔除低质量设计。

依赖: biotite (可选, 用于高效 PDB 解析), Bio.PDB (fallback), scipy, numpy

使用方式:
    from inverse_optimize.quality_filter import filter_pdb_by_quality

    result = filter_pdb_by_quality(
        pdb_path="/path/to/design.pdb",
        min_iptm=0.5,
        min_binder_plddt=0.7,
        binder_chain="B",  # binder 链名必须显式提供 (无默认链假设)
    )
    if result["passed"]:
        print(f"Quality OK: iPTM={result['iptm']:.3f}")
    else:
        print(f"Rejected: {result['reasons']}")
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---- Data structures ----


@dataclass
class QualityFilterConfig:
    """质量过滤阈值配置。

    Attributes
    ----------
    min_iptm : float
        最低 iPTM (interface TM-score), 默认 0.50。
    min_ptm : float | None
        最低 pTM (global TM-score), None 表示不检查。
    min_pair_chains_iptm : float | None
        最低 binder-target pair ipTM (REMARK 键 ``pair_chains_iptm``,
        t2b 方向残基加权合并值, 2026/10/19 新增), None 表示不检查。
        与 ``min_iptm`` 互为指标源开关的两翼 —— core.py 按
        ``PipelineConfig.iptm_metric`` 只给其中之一喂阈值。
    min_binder_plddt : float | None
        Binder 链 CA 原子平均 pLDDT 下限, None 表示不检查。
    min_distogram_binding_confidence : float | None
        最低 distogram 结合置信度, None 表示不检查。
    max_ca_clashes : int
        最大允许的 CA 原子冲突数, 默认 0。
    ca_clash_cutoff : float
        CA 冲突距离阈值 (Å), 默认 2.5。
    binder_chain : str
        PDB 中 binder 的链 ID, 默认 "" (必填 — 无默认链假设;
        设置 min_binder_plddt 时若为空会触发 assert)。
    """

    min_iptm: float = 0.50
    min_ptm: float | None = None
    ##### claude 2026/10/19 start #####
    min_pair_chains_iptm: float | None = None
    ##### claude 2026/10/19 end #####
    min_binder_plddt: float | None = 0.70
    min_distogram_binding_confidence: float | None = None
    max_ca_clashes: int = 0
    ca_clash_cutoff: float = 2.5
    ##### claude 2026/8/13 start #####
    # 无默认链假设: 多链 target 时 binder 链名不再是固定的 "B",
    # 必须由调用方显式提供 (留空 + min_binder_plddt 检查会触发 assert)。
    binder_chain: str = ""
    ##### claude 2026/8/13 end #####


# ---- Remark parsing ----


def _parse_remarks(pdb_path: str) -> dict[str, float | None]:
    """从 PDB REMARK 250 行提取质量指标。

    期望格式 (由 ``pdb_with_quality_remarks`` 生成)::

        REMARK 250 PREDICTED QUALITY METRICS
        REMARK 250  iPTM=0.7532
        REMARK 250  pTM=0.8210
        REMARK 250  distogram_binding_confidence=0.6815
        REMARK 250  pair_chains_iptm=0.6123
        REMARK 250  pair_chains_iptm_b2t=0.5891

    Returns
    -------
    dict
        {"iptm": float|None, "ptm": float|None,
         "distogram_binding_confidence": float|None,
         "pair_chains_iptm": float|None, "pair_chains_iptm_b2t": float|None}
    """
    metrics: dict[str, float | None] = {
        "iptm": None,
        "ptm": None,
        "distogram_binding_confidence": None,
        ##### claude 2026/10/19 start #####
        "pair_chains_iptm": None,
        "pair_chains_iptm_b2t": None,
        ##### claude 2026/10/19 end #####
    }
    # Map REMARK key to dict key
    key_map = {
        "iPTM": "iptm",
        "pTM": "ptm",
        "distogram_binding_confidence": "distogram_binding_confidence",
        ##### claude 2026/10/19 start #####
        "pair_chains_iptm": "pair_chains_iptm",
        "pair_chains_iptm_b2t": "pair_chains_iptm_b2t",
        ##### claude 2026/10/19 end #####
    }

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("REMARK 250"):
                continue
            # Try the key=value pattern
            m = re.search(r"REMARK\s+250\s+(\S+)=([\d.]+)", line)
            if m:
                remark_key = m.group(1)
                if remark_key in key_map:
                    try:
                        metrics[key_map[remark_key]] = float(m.group(2))
                    except ValueError:
                        pass

    return metrics


# ---- B-factor pLDDT extraction ----


# PDB column indices for ATOM records (1-indexed within the line)
_PDB_ATOM_CHAIN_COL = 22  # 0-indexed: 21
_PDB_ATOM_NAME_COL_START = 13  # 0-indexed: 12
_PDB_ATOM_NAME_COL_END = 16  # 0-indexed: 15
_PDB_ATOM_BFACTOR_COL_START = 61  # 0-indexed: 60
_PDB_ATOM_BFACTOR_COL_END = 66  # 0-indexed: 65

# AlphaFold convention: B-factor = pLDDT × 100
_PLDDT_B_FACTOR_SCALE = 100.0


def _plddt_from_bfactor_line(line: str) -> tuple[str, str, float]:
    """Parse one ATOM line: return (chain_id, atom_name, pLDDT)."""
    chain_id = line[21]
    atom_name = line[12:16].strip()
    bfactor_str = line[60:66].strip()
    plddt = float(bfactor_str) / _PLDDT_B_FACTOR_SCALE if bfactor_str else 0.0
    return chain_id, atom_name, plddt


##### claude 2026/8/13 start #####
def _average_binder_plddt(pdb_path: str, binder_chain: str) -> float:
##### claude 2026/8/13 end #####
    """从 PDB B-factor 列计算 binder 链 CA 原子的平均 pLDDT。

    Parameters
    ----------
    pdb_path : str
        PDB 文件路径。
    binder_chain : str
        Binder 链 ID。

    Returns
    -------
    float
        平均 pLDDT (0-1 范围)。若链不存在则返回 0.0。
    """
    ca_plddts: list[float] = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            chain_id, atom_name, plddt = _plddt_from_bfactor_line(line)
            if chain_id == binder_chain and atom_name == "CA":
                ca_plddts.append(plddt)

    if not ca_plddts:
        return 0.0
    return float(np.mean(ca_plddts))


# ---- CA clash detection (fast, no external deps beyond scipy) ----


def _count_ca_clashes(
    pdb_path: str,
    cutoff: float = 2.5,
) -> int:
    """用 scipy.spatial.cKDTree 统计 CA 原子冲突数。

    排除同残基内、同链相邻残基 (|i-j|≤1) 的 CA 对。
    """
    from scipy.spatial import cKDTree

    ca_coords: list[np.ndarray] = []
    ca_info: list[tuple[str, int]] = []  # (chain_id, res_id)

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chain_id = line[21]
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                continue
            ca_coords.append(np.array([x, y, z]))
            ca_info.append((chain_id, res_id))

    if len(ca_coords) < 2:
        return 0

    tree = cKDTree(np.array(ca_coords))
    pairs = tree.query_pairs(cutoff)

    clashes = 0
    for i, j in pairs:
        chain_i, res_i = ca_info[i]
        chain_j, res_j = ca_info[j]
        # Same residue
        if chain_i == chain_j and res_i == res_j:
            continue
        # Adjacent residues on same chain
        if chain_i == chain_j and abs(res_i - res_j) <= 1:
            continue
        clashes += 1

    return clashes


# ---- Main filter function ----


def filter_pdb_by_quality(
    pdb_path: str,
    config: QualityFilterConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """基于 PDB 内嵌质量指标对单个设计进行预过滤。

    按顺序检查: PDB 存在性 → iPTM → pTM → binder pLDDT →
    distogram_binding_confidence → CA clashes。
    任一条件不满足即标记为 failed 并记录原因。

    Parameters
    ----------
    pdb_path : str
        PDB 文件路径 (ESMFold2 Hero Critic 输出的 annotated PDB)。
    config : QualityFilterConfig | None
        过滤配置。None 则使用默认值。
    **kwargs
        可覆盖 config 中的任意字段, 例如 ``min_iptm=0.6``。

    Returns
    -------
    dict
        包含:
        - "passed": bool — 是否通过所有过滤
        - "pdb_path": str — 输入的 PDB 路径
        - "iptm": float|None — 解析到的 iPTM
        - "ptm": float|None — 解析到的 pTM
        - "binder_plddt": float — binder 链 CA 平均 pLDDT
        - "distogram_binding_confidence": float|None
        - "pair_chains_iptm": float|None — binder-target pair ipTM (t2b 合并)
        - "pair_chains_iptm_b2t": float|None — 反方向合并值 (仅诊断)
        - "ca_clashes": int — CA 冲突数
        - "reasons": list[str] — 未通过的原因 (pass 时为空)
    """
    if config is None:
        config = QualityFilterConfig()

    # Apply keyword overrides
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)

    reasons: list[str] = []

    # 0. File existence
    if not os.path.isfile(pdb_path):
        return {
            "passed": False,
            "pdb_path": pdb_path,
            "iptm": None,
            "ptm": None,
            "binder_plddt": 0.0,
            "distogram_binding_confidence": None,
            ##### claude 2026/10/19 start #####
            "pair_chains_iptm": None,
            "pair_chains_iptm_b2t": None,
            ##### claude 2026/10/19 end #####
            "ca_clashes": -1,
            "reasons": [f"PDB file not found: {pdb_path}"],
        }

    # 1. Parse REMARK metrics
    remarks = _parse_remarks(pdb_path)
    iptm = remarks["iptm"]
    ptm = remarks["ptm"]
    dist_binding_conf = remarks["distogram_binding_confidence"]
    ##### claude 2026/10/19 start #####
    pair_iptm = remarks["pair_chains_iptm"]
    ##### claude 2026/10/19 end #####

    # 2. Check iPTM
    if config.min_iptm is not None:
        if iptm is None:
            reasons.append(
                f"iPTM not found in REMARK records (need {config.min_iptm:.2f})"
            )
        elif iptm < config.min_iptm:
            reasons.append(
                f"iPTM={iptm:.4f} < {config.min_iptm:.2f}"
            )

    ##### claude 2026/10/19 start #####
    # 2b. Binder-target pair ipTM (t2b 残基加权合并值; 与 min_iptm 互为
    # 指标源开关, core.py 按 PipelineConfig.iptm_metric 二选一喂阈值)。
    if config.min_pair_chains_iptm is not None:
        if pair_iptm is None:
            reasons.append(
                "pair_chains_iptm not found in REMARK records "
                f"(need {config.min_pair_chains_iptm:.2f})"
            )
        elif pair_iptm < config.min_pair_chains_iptm:
            reasons.append(
                f"pair_chains_iptm={pair_iptm:.4f} "
                f"< {config.min_pair_chains_iptm:.2f}"
            )
    ##### claude 2026/10/19 end #####

    # 3. Check pTM
    if config.min_ptm is not None:
        if ptm is None:
            reasons.append(
                f"pTM not found in REMARK records (need {config.min_ptm:.2f})"
            )
        elif ptm < config.min_ptm:
            reasons.append(
                f"pTM={ptm:.4f} < {config.min_ptm:.2f}"
            )

    # 4. Binder pLDDT
    ##### claude 2026/8/13 start #####
    # binder 链名必须由调用方提供 (无默认链假设)。若调用方未提供且需要
    # 检查 binder pLDDT, 直接报错而不是静默测量错误链。
    if config.min_binder_plddt is not None:
        assert config.binder_chain, (
            "binder_chain must be provided (no default chain assumption) "
            "when min_binder_plddt is set."
        )
        binder_plddt = _average_binder_plddt(pdb_path, config.binder_chain)
        if binder_plddt < config.min_binder_plddt:
            reasons.append(
                f"Binder pLDDT={binder_plddt:.4f} < {config.min_binder_plddt:.2f}"
            )
    else:
        binder_plddt = 0.0
    ##### claude 2026/8/13 end #####

    # 5. Distogram binding confidence
    if config.min_distogram_binding_confidence is not None:
        if dist_binding_conf is None:
            reasons.append(
                "distogram_binding_confidence not found in REMARK records "
                f"(need {config.min_distogram_binding_confidence:.2f})"
            )
        elif dist_binding_conf < config.min_distogram_binding_confidence:
            reasons.append(
                f"dist_binding_confidence={dist_binding_conf:.4f} "
                f"< {config.min_distogram_binding_confidence:.2f}"
            )

    # 6. CA clashes
    ca_clashes = _count_ca_clashes(pdb_path, config.ca_clash_cutoff)
    if ca_clashes > config.max_ca_clashes:
        reasons.append(
            f"CA clashes={ca_clashes} > {config.max_ca_clashes}"
        )

    return {
        "passed": len(reasons) == 0,
        "pdb_path": pdb_path,
        "iptm": iptm,
        "ptm": ptm,
        "binder_plddt": binder_plddt,
        "distogram_binding_confidence": dist_binding_conf,
        ##### claude 2026/10/19 start #####
        "pair_chains_iptm": pair_iptm,
        "pair_chains_iptm_b2t": remarks["pair_chains_iptm_b2t"],
        ##### claude 2026/10/19 end #####
        "ca_clashes": ca_clashes,
        "reasons": reasons,
    }


def batch_filter_pdbs(
    pdb_paths: list[str],
    config: QualityFilterConfig | None = None,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """批量质量过滤。

    Parameters
    ----------
    pdb_paths : list[str]
        候选 PDB 路径列表。
    config : QualityFilterConfig | None
    **kwargs
        传递到 ``filter_pdb_by_quality``。

    Returns
    -------
    passed : list[dict]
    failed : list[dict]
    """
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for pdb_path in pdb_paths:
        result = filter_pdb_by_quality(pdb_path, config=config, **kwargs)
        if result["passed"]:
            passed.append(result)
        else:
            failed.append(result)

    return passed, failed
