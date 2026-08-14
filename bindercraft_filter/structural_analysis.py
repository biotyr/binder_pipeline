"""
结构分析工具 — 移植自 BindCraft

原始文件: /home/biotyr/esmbind/BindCraft-main/functions/biopython_utils.py
依赖: Bio (biopython), scipy, numpy, DSSP 可执行文件
"""

##### claude 2026/8/13 start #####
from __future__ import annotations  # 支持 list[str] | str 等现代注解 (py3.8+)
##### claude 2026/8/13 end #####

import math
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, DSSP, Selection, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# 三字母 → 单字母氨基酸映射
THREE_TO_ONE: dict[str, str] = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def calculate_clash_score(
    pdb_file: str,
    threshold: float = 2.4,
    only_ca: bool = False,
) -> int:
    """cKDTree 原子冲突检测 — 移植自 BindCraft biopython_utils.py:88-128

    Parameters
    ----------
    pdb_file : str
        PDB 文件路径
    threshold : float
        冲突距离阈值 (Å), CA 模式 2.5, 全原子 2.4
    only_ca : bool
        True=仅 CA 原子 (快速筛查 trajectory 质量),
        False=全非氢原子 (链间全原子冲突)

    Returns
    -------
    int
        有效冲突对数
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms: list[np.ndarray] = []
    atom_info: list[tuple[str, int, str, np.ndarray]] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((
                        chain.id, residue.id[1], atom.get_name(), atom.coord
                    ))

    if len(atoms) < 2:
        return 0

    tree = cKDTree(np.array(atoms))
    pairs = tree.query_pairs(threshold)

    valid_pairs: set[tuple[int, int]] = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # 排除同残基内
        if chain_i == chain_j and res_i == res_j:
            continue
        # 排除同链相邻残基
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue
        # 侧链冲突检测: 仅跨链
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)


def hotspot_residues(
    pdb_file: str,
    binder_chain: str = "B",
    target_chains: list[str] | str = "A",
    atom_distance_cutoff: float = 4.0,
) -> dict[int, str]:
    """cKDTree 界面残基检测 — 移植自 BindCraft biopython_utils.py:138-173

    查找与 target 链在 cutoff 距离内有原子接触的 binder 残基。

    Parameters
    ----------
    pdb_file : str
        复合体 PDB
    binder_chain : str
        Binder 链 ID (默认 "B")
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str
        Target 链 ID (默认 "A")。多链 target 时传入链名列表,
        检测 binder 与全部 target 链原子并集的界面接触。
        binder_chain 自身不得出现在列表中。
    ##### claude 2026/8/13 end #####
    atom_distance_cutoff : float
        接触距离阈值 (Å)

    Returns
    -------
    dict
        {PDB_residue_number: single_letter_AA_code} 界面残基映射
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)

    binder_atoms = list(Selection.unfold_entities(structure[0][binder_chain], 'A'))
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    ##### claude 2026/8/13 start #####
    # 多链 target: 全部 target 链原子并集, 单次 cKDTree 查询
    # (与逐链查询后合并数学等价)。排除 binder 自身, 防止链内
    # 接触被误标为界面残基。
    labels = [target_chains] if isinstance(target_chains, str) else list(target_chains)
    assert binder_chain not in labels, (
        f"binder_chain {binder_chain!r} must not appear in target_chains."
    )
    target_atoms: list = []
    for label in labels:
        target_atoms.extend(Selection.unfold_entities(structure[0][label], 'A'))
    target_coords = np.array([atom.coord for atom in target_atoms])
    ##### claude 2026/8/13 end #####

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    interacting_residues: dict[int, str] = {}
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        if binder_resname in THREE_TO_ONE:
            aa_single_letter = THREE_TO_ONE[binder_resname]
            for close_idx in close_indices:
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


def calc_ss_percentage(
    pdb_file: str,
    dssp_path: str,
    chain_id: str,
    ##### claude 2026/8/13 start #####
    target_chains: list[str] | str = "A",
    ##### claude 2026/8/13 end #####
    atom_distance_cutoff: float = 4.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """DSSP 二级结构分析 — 移植自 BindCraft biopython_utils.py:176-235

    同时分析 binder 全链和界面残基的二级结构分布,
    并从 PDB B-factor 列读取 pLDDT 值。

    Parameters
    ----------
    pdb_file : str
        复合体 PDB (含 B-factor=pLDDT)
    dssp_path : str
        DSSP 可执行文件路径
    chain_id : str
        Binder 链 ID, 必须显式提供 (无默认链假设)。
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str
        Target 链 ID (单链 "A" 或多链 ["A", "B"]), 用于界面残基检测。
    ##### claude 2026/8/13 end #####
    atom_distance_cutoff : float
        界面残基检测距离

    Returns
    -------
    tuple[float × 8]
        (binder_helix%, binder_sheet%, binder_loop%,
         interface_helix%, interface_sheet%, interface_loop%,
         i_pLDDT, ss_pLDDT)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    dssp = DSSP(model, pdb_file, dssp=dssp_path)

    ss_counts: dict[str, int] = defaultdict(int)
    ss_interface_counts: dict[str, int] = defaultdict(int)
    plddts_interface: list[float] = []
    plddts_ss: list[float] = []

    chain = model[chain_id]
    ##### claude 2026/8/13 start #####
    interacting_residues = set(
        hotspot_residues(
            pdb_file, chain_id,
            target_chains=target_chains,
            atom_distance_cutoff=atom_distance_cutoff,
        ).keys()
    )
    ##### claude 2026/8/13 end #####

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) not in dssp:
            continue
        ss = dssp[(chain_id, residue_id)][2]  # DSSP 二级结构代码

        ss_type = 'loop'
        if ss in ('H', 'G', 'I'):
            ss_type = 'helix'
        elif ss == 'E':
            ss_type = 'sheet'

        ss_counts[ss_type] += 1

        if ss_type != 'loop':
            avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
            plddts_ss.append(avg_plddt_ss)

        if residue_id in interacting_residues:
            ss_interface_counts[ss_type] += 1
            avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
            plddts_interface.append(avg_plddt_residue)

    total_residues = sum(ss_counts.values())
    total_interface = sum(ss_interface_counts.values())

    h_pct, s_pct, l_pct = _calculate_percentages(
        total_residues, ss_counts['helix'], ss_counts['sheet']
    )
    ih_pct, is_pct, il_pct = _calculate_percentages(
        total_interface, ss_interface_counts['helix'], ss_interface_counts['sheet']
    )

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0.0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0.0

    return h_pct, s_pct, l_pct, ih_pct, is_pct, il_pct, i_plddt, ss_plddt


def _calculate_percentages(
    total: int, helix: int, sheet: int
) -> tuple[float, float, float]:
    """计算二级结构百分比。"""
    if total == 0:
        return 0.0, 0.0, 0.0
    h = round((helix / total) * 100, 2)
    s = round((sheet / total) * 100, 2)
    l = round(((total - helix - sheet) / total) * 100, 2)
    return h, s, l


def validate_design_sequence(
    sequence: str,
    num_clashes: int,
    omit_AAs: str | None = None,
) -> str:
    """序列质量验证 — 移植自 BindCraft biopython_utils.py:17-46

    检查: 冲突数、禁用氨基酸、UV 吸收 (280nm)

    Parameters
    ----------
    sequence : str
        单字母氨基酸序列
    num_clashes : int
        弛豫后冲突数
    omit_AAs : str | None
        逗号分隔的禁用氨基酸, 如 "C,M"

    Returns
    -------
    str
        空格分隔的警告信息; 若无问题则返回空字符串
    """
    note_array: list[str] = []

    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    if omit_AAs:
        restricted_AAs = omit_AAs.split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA.strip() in sequence:
                note_array.append(f'Contains:{restricted_AA.strip()}!')

    analysis = ProteinAnalysis(sequence)
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_1pct = round(
        extinction_coefficient_reduced / molecular_weight * 0.01, 2
    )

    if extinction_1pct <= 2:
        note_array.append(
            f'Absorption value is {extinction_1pct}, '
            'consider adding tryptophane to design.'
        )

    return ' '.join(note_array)


def target_pdb_rmsd(
    trajectory_pdb: str, starting_pdb: str, chain_ids_string: str,
) -> float:
    """Target RMSD 计算 — 移植自 BindCraft biopython_utils.py:49-85

    用 Bio.PDB.Superimposer 叠加两个结构的 target chain 后计算 RMSD。

    Parameters
    ----------
    trajectory_pdb : str
        预测的复合体 PDB (target 在 chain A)
    starting_pdb : str
        输入的目标蛋白 PDB
    chain_ids_string : str
        逗号分隔的 target chain ID, 如 "A" 或 "A,B"

    Returns
    -------
    float
        叠加后 RMSD (Å)
    """
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)

    chain_trajectory = structure_trajectory[0]['A']
    chain_ids = chain_ids_string.split(',')

    residues_starting: list = []
    for chain_id in chain_ids:
        chain = structure_starting[0][chain_id.strip()]
        residues_starting.extend(
            residue for residue in chain if is_aa(residue, standard=True)
        )

    residues_trajectory = [
        residue for residue in chain_trajectory if is_aa(residue, standard=True)
    ]

    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]

    atoms_starting = [
        residue['CA'] for residue in residues_starting if 'CA' in residue
    ]
    atoms_trajectory = [
        residue['CA'] for residue in residues_trajectory if 'CA' in residue
    ]

    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    return round(sup.rms, 2)
