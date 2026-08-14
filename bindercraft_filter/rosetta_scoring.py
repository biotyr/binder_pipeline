"""
Rosetta 接口评分与结构弛豫 — 移植自 BindCraft

原始文件: /home/biotyr/esmbind/BindCraft-main/functions/pyrosetta_utils.py
依赖: pyrosetta (需独立安装 + 商业许可), hotspot_residues (同包 structural_analysis)
"""

##### claude 2026/8/13 start #####
from __future__ import annotations  # 支持 list[str] | str 等现代注解 (py3.8+)
##### claude 2026/8/13 end #####

import os
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

from .structural_analysis import hotspot_residues


def clean_pdb(pdb_file: str) -> None:
    """清理 Rosetta 输出的多余行,仅保留 ATOM/HETATM/MODEL/TER/END/LINK。
    原始: BindCraft generic_utils.py:303-310"""
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [
            line for line in f_in
            if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK'))
        ]
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)


def score_interface(
    pdb_file: str,
    binder_chain: str,
    target_chains: list[str] | str = "A",
) -> tuple[dict, dict, str]:
    """Rosetta InterfaceAnalyzer 界面评分 — 移植自 BindCraft pyrosetta_utils.py:20-144

    Parameters
    ----------
    pdb_file : str
        复合体 PDB 文件路径 (需包含 target 链 + binder 链)
    binder_chain : str
        Binder 链 ID, 必须显式提供 (无默认链假设)。
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str
        Target 链 ID (单链 "A" 或多链 ["A", "B"])。
        set_interface 定义为 target 组对 binder: 如 "AB_C"。
    ##### claude 2026/8/13 end #####

    Returns
    -------
    interface_scores : dict
        {
            'binder_score': float,               # binder 单体 Rosetta 总能量 (REU)
            'surface_hydrophobicity': float,      # binder 表面疏水残基比例 (0-1)
            'interface_sc': float,                # 形状互补性 Sc (0-1)
            'interface_packstat': float,          # 界面堆积密度 (0-1)
            'interface_dG': float,                # 界面结合自由能 (REU)
            'interface_dSASA': float,             # 界面溶剂可及表面积变化
            'interface_dG_SASA_ratio': float,     # dG/dSASA×100
            'interface_fraction': float,          # 界面占 binder 表面积百分比
            'interface_hydrophobicity': float,    # 界面疏水残基占比
            'interface_nres': int,                # 界面残基数
            'interface_interface_hbonds': int,    # 界面跨链氢键数
            'interface_hbond_percentage': float,  # 氢键/界面残基数×100
            'interface_delta_unsat_hbonds': int,  # 界面埋藏未满足氢键数
            'interface_delta_unsat_hbonds_percentage': float,
        }
    interface_AA : dict
        {aa: count} 界面各氨基酸计数, 20 种标准氨基酸
    interface_residues_str : str
        逗号分隔的界面残基 ID, 如 "B5,B12,B34"
    """
    # 加载 pose (pyrosetta_utils.py:22)
    pose = pr.pose_from_pdb(pdb_file)

    ##### claude 2026/8/13 start #####
    target_labels = [target_chains] if isinstance(target_chains, str) else list(target_chains)
    assert binder_chain not in target_labels, (
        f"binder_chain {binder_chain!r} must not appear in target_chains."
    )
    ##### claude 2026/8/13 end #####

    # 配置 InterfaceAnalyzerMover (pyrosetta_utils.py:25-35)
    iam = InterfaceAnalyzerMover()
    ##### claude 2026/8/13 start #####
    # 多链 target: 接口定义为 target 组对 binder (如 "AB_C")。
    # Rosetta 原生支持下划线两侧连写多链; A-B 自身接触不计入界面。
    iam.set_interface(f"{''.join(target_labels)}_{binder_chain}")
    ##### claude 2026/8/13 end #####
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # 界面氨基酸组成 (pyrosetta_utils.py:38-56)
    interface_AA: dict[str, int] = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    ##### claude 2026/8/13 start #####
    interface_residues_set = hotspot_residues(
        pdb_file, binder_chain, target_chains=target_labels
    )
    ##### claude 2026/8/13 end #####
    interface_residues_pdb_ids: list[str] = []

    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_str = ','.join(interface_residues_pdb_ids)

    # 界面疏水性: 疏水残基 ACFILMPVWY (pyrosetta_utils.py:59-63)
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    interface_hydrophobicity = (
        (hydrophobic_count / interface_nres) * 100 if interface_nres else 0.0
    )

    # 提取 Rosetta 指标 (pyrosetta_utils.py:66-82)
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100

    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" '
        'scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" '
        'dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    interface_hbond_percentage = (
        (interface_interface_hbonds / interface_nres) * 100 if interface_nres else None
    )
    interface_bunsch_percentage = (
        (interface_delta_unsat_hbonds / interface_nres) * 100 if interface_nres else None
    )

    # Binder 单体能量 (pyrosetta_utils.py:85-89)
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # Binder SASA 分数 (pyrosetta_utils.py:91-94)
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    interface_binder_fraction = (
        (interface_dSASA / binder_sasa) * 100 if binder_sasa > 0 else 0.0
    )

    # 表面疏水性 (pyrosetta_utils.py:101-120)
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0
    for i in range(1, len(surface_res) + 1):
        if surface_res[i]:
            res = binder_pose.residue(i)
            if res.is_apolar() or res.name() in ('PHE', 'TRP', 'TYR'):
                exp_apol_count += 1
            total_count += 1
    surface_hydrophobicity = exp_apol_count / total_count if total_count > 0 else 0.0

    # 组装输出 (pyrosetta_utils.py:124-144)
    interface_scores = {
        'binder_score': binder_score,
        'surface_hydrophobicity': surface_hydrophobicity,
        'interface_sc': interface_sc,
        'interface_packstat': interface_packstat,
        'interface_dG': interface_dG,
        'interface_dSASA': interface_dSASA,
        'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
        'interface_fraction': interface_binder_fraction,
        'interface_hydrophobicity': interface_hydrophobicity,
        'interface_nres': interface_nres,
        'interface_interface_hbonds': interface_interface_hbonds,
        'interface_hbond_percentage': interface_hbond_percentage,
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
        'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage,
    }

    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores, interface_AA, interface_residues_str


def pr_relax(
    pdb_file: str,
    relaxed_pdb_path: str,
    enable_bb_movement: bool = True,
    max_iter: int = 200,
) -> None:
    """PyRosetta FastRelax 结构弛豫 — 移植自 BindCraft pyrosetta_utils.py:204-243

    执行 backbone + sidechain FastRelax,输出对齐到原始结构的弛豫 PDB,
    并将原始 B-factor (pLDDT) 复制到弛豫结构中。

    Parameters
    ----------
    pdb_file : str
        输入 PDB 路径
    relaxed_pdb_path : str
        弛豫后输出 PDB 路径
    enable_bb_movement : bool
        是否允许 backbone 移动 (默认 True; False 可加速 30% 但指标略差)
    max_iter : int
        FastRelax 最大迭代数 (默认 200)
    """
    if os.path.exists(relaxed_pdb_path):
        return

    pose = pr.pose_from_pdb(pdb_file)
    start_pose = pose.clone()

    # MoveMap 配置 (pyrosetta_utils.py:211-214)
    mmf = MoveMap()
    mmf.set_chi(True)              # 允许侧链移动
    mmf.set_bb(enable_bb_movement) # 允许 backbone 移动
    mmf.set_jump(False)            # 禁止整链平移

    # FastRelax 配置 (pyrosetta_utils.py:216-224)
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf)
    fastrelax.max_iter(max_iter)
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    # 对齐到原始结构 (pyrosetta_utils.py:227-231)
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    # 复制 B-factor 从原始到弛豫结构 (pyrosetta_utils.py:233-239)
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    pose.dump_pdb(relaxed_pdb_path)
    clean_pdb(relaxed_pdb_path)


def align_pdbs(
    reference_pdb: str, align_pdb: str,
    reference_chain_id: str, align_chain_id: str,
) -> None:
    """将 align_pdb 的指定链对齐叠加到 reference_pdb 的指定链 — 移植自 pyrosetta_utils.py:147-169

    原地修改 align_pdb 文件。
    """
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    aln = AlignChainMover()
    aln.pose(reference_pose)

    reference_chain_id = reference_chain_id.split(',')[0]
    align_chain_id = align_chain_id.split(',')[0]

    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        reference_chain_id, reference_pose
    )
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(
        align_chain_id, align_pose
    )

    aln.source_chain(align_chain)
    aln.target_chain(reference_chain)
    aln.apply(align_pose)

    align_pose.dump_pdb(align_pdb)
    clean_pdb(align_pdb)


def unaligned_rmsd(
    reference_pdb: str, target_pdb: str,
    reference_chain_id: str, target_chain_id: str,
) -> float:
    """计算两个结构中指定链的未对齐 RMSD (不做叠加) — 移植自 pyrosetta_utils.py:172-201

    用途: 评估 binder 在"原始 trajectory"和"重预测"之间的位置差异。
    """
    reference_pose = pr.pose_from_pdb(reference_pdb)
    target_pose = pr.pose_from_pdb(target_pdb)

    reference_chain_selector = ChainSelector(reference_chain_id)
    target_chain_selector = ChainSelector(target_chain_id)

    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    target_chain_subset = target_chain_selector.apply(target_pose)

    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    target_residue_indices = get_residues_from_subset(target_chain_subset)

    reference_chain_pose = pr.Pose()
    target_chain_pose = pr.Pose()

    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(target_chain_pose, target_pose, target_residue_indices)

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(target_chain_pose)

    return round(rmsd, 2)
