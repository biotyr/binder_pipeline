"""
BindCraft 过滤打分模块 — 移植到 ESMFold2 Pipeline 后处理阶段

来源: /home/biotyr/esmbind/BindCraft-main/functions/
   pyrosetta_utils.py   → rosetta_scoring.py    (Rosetta 接口分析与弛豫)
   biopython_utils.py   → structural_analysis.py (冲突检测/DSSP/界面残基/序列验证)
   generic_utils.py     → filter_check.py        (阈值过滤逻辑)

用途: 对 ESMFold2 Hero Critic 输出的全原子 PDB 做物理化学验证,
     补充 ESMFold2 原生缺失的力场级质量评估。
"""

from .rosetta_scoring import (
    score_interface,
    pr_relax,
    align_pdbs,
    unaligned_rmsd,
)

from .structural_analysis import (
    calculate_clash_score,
    hotspot_residues,
    calc_ss_percentage,
    validate_design_sequence,
    target_pdb_rmsd,
)

from .filter_check import (
    check_filters,
    generate_filter_pass_csv,
    update_failures,
    calculate_averages,
)

__all__ = [
    # Rosetta
    "score_interface",
    "pr_relax",
    "align_pdbs",
    "unaligned_rmsd",
    # Structural
    "calculate_clash_score",
    "hotspot_residues",
    "calc_ss_percentage",
    "validate_design_sequence",
    "target_pdb_rmsd",
    # Filter
    "check_filters",
    "generate_filter_pass_csv",
    "update_failures",
    "calculate_averages",
]
