"""
ESMFold2 + BindCraft 过滤集成模块

将 BindCraft 的 Rosetta 物理评分接入 ESMFold2 Critic 输出,
对 Hero Critic 产生的全原子 PDB 执行完整的物理化学评估。

使用方式:
    from bindercraft_filter.integration import screen_critic_results

    passed, failed = screen_critic_results(
        critic_results,        # ESMFold2 design_binder() 返回的 critic_results
        dssp_path="/path/to/dssp",
        filters_json="./default_filters.json",
        relax_first=True,
    )
"""

##### claude 2026/8/13 start #####
from __future__ import annotations  # 支持 list[str] | str 等现代注解 (py3.8+)
##### claude 2026/8/13 end #####

import json
import os
from typing import Any

from .rosetta_scoring import score_interface, pr_relax
from .structural_analysis import (
    calculate_clash_score,
    calc_ss_percentage,
    validate_design_sequence,
)
from .filter_check import check_filters


def screen_single_pdb(
    pdb_path: str,
    binder_sequence: str,
    dssp_path: str,
    binder_chain: str,
    target_chains: list[str] | str = "A",
    relax_first: bool = True,
    relaxed_output_dir: str | None = None,
    omit_AAs: str | None = None,
    af2_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    """对单个 PDB 运行 BindCraft 全套物理评分。

    Parameters
    ----------
    pdb_path : str
        复合体 PDB 路径 (ESMFold2 Hero Critic 输出)
    binder_sequence : str
        Binder 氨基酸序列 (单字母)
    dssp_path : str
        DSSP 可执行文件路径
    binder_chain : str
        Binder 链 ID, 必须显式提供 (无默认链假设)。
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str
        Target 链 ID (单链 "A" 或多链 ["A", "B"])。
        用于 Rosetta set_interface ("AB_C") 与界面残基检测。
    ##### claude 2026/8/13 end #####
    relax_first : bool
        是否先执行 Rosetta FastRelax (推荐 True)
    relaxed_output_dir : str | None
        弛豫后 PDB 保存目录 (None = 不保存)
    omit_AAs : str | None
        逗号分隔的禁用氨基酸
    af2_confidence : dict | None
        ESMFold2 已有的置信度指标, 例如:
        {"plddt": 0.85, "ptm": 0.72, "i_ptm": 0.65, "pae": 3.2, "i_pae": 2.1}

    Returns
    -------
    dict
        完整评分字典, 包含:
        - interface_scores: Rosetta 界面分析结果
        - clash_scores: 冲突检测 (弛豫前后)
        - ss_percentages: 二级结构分布
        - sequence_notes: 序列质量警告
        - af2_confidence: 传入的置信度指标
    """
    results: dict[str, Any] = {
        "pdb_path": pdb_path,
        "binder_sequence": binder_sequence,
    }

    # 1. 弛豫前冲突检测
    clashes_before = calculate_clash_score(pdb_path, threshold=2.4, only_ca=False)
    ca_clashes = calculate_clash_score(pdb_path, threshold=2.5, only_ca=True)
    results["clashes_before_relax"] = clashes_before
    results["ca_clashes"] = ca_clashes

    # 2. Rosetta FastRelax (可选)
    relaxed_pdb = pdb_path
    if relax_first:
        if relaxed_output_dir:
            os.makedirs(relaxed_output_dir, exist_ok=True)
            relaxed_pdb = os.path.join(
                relaxed_output_dir,
                os.path.basename(pdb_path).replace('.pdb', '_relaxed.pdb')
            )
        else:
            relaxed_pdb = pdb_path.replace('.pdb', '_relaxed.pdb')
        pr_relax(pdb_path, relaxed_pdb)
    results["relaxed_pdb"] = relaxed_pdb

    # 3. 弛豫后冲突检测
    clashes_after = calculate_clash_score(relaxed_pdb, threshold=2.4, only_ca=False)
    results["clashes_after_relax"] = clashes_after

    # 4. Rosetta 界面分析
    ##### claude 2026/8/13 start #####
    interface_scores, interface_AA, interface_residues = score_interface(
        relaxed_pdb,
        binder_chain=binder_chain,
        target_chains=target_chains,
    )
    ##### claude 2026/8/13 end #####
    results["interface_scores"] = interface_scores
    results["interface_AA"] = interface_AA
    results["interface_residues"] = interface_residues

    # 5. DSSP 二级结构
    ##### claude 2026/8/13 start #####
    ss_result = calc_ss_percentage(
        pdb_path, dssp_path,
        chain_id=binder_chain,
        target_chains=target_chains,
    )
    ##### claude 2026/8/13 end #####
    (
        binder_helix, binder_sheet, binder_loop,
        interface_helix, interface_sheet, interface_loop,
        i_plddt, ss_plddt,
    ) = ss_result
    results["ss_percentages"] = {
        "Binder_Helix%": binder_helix,
        "Binder_BetaSheet%": binder_sheet,
        "Binder_Loop%": binder_loop,
        "Interface_Helix%": interface_helix,
        "Interface_BetaSheet%": interface_sheet,
        "Interface_Loop%": interface_loop,
    }
    results["i_pLDDT"] = i_plddt
    results["ss_pLDDT"] = ss_plddt

    # 6. 序列质量
    seq_notes = validate_design_sequence(
        binder_sequence, clashes_after, omit_AAs=omit_AAs
    )
    results["sequence_notes"] = seq_notes

    # 7. 传入的 AF2 置信度
    if af2_confidence:
        results["af2_confidence"] = af2_confidence

    return results


def build_filter_data_dict(
    screening_result: dict[str, Any],
    af2_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    """从 screening_result 构建 check_filters() 所需的 data_dict。

    将 Rosetta 指标 + 结构指标 + AF2 置信度指标统一组装,
    支持 Average_ 和 per-model 前缀。

    Parameters
    ----------
    screening_result : dict
        screen_single_pdb() 的输出
    af2_confidence : dict | None
        额外的 AF2/ESMFold2 置信度指标

    Returns
    -------
    dict
        可直接传入 check_filters(data_dict, filters) 的字典
    """
    iface = screening_result["interface_scores"]
    ss = screening_result["ss_percentages"]

    data: dict[str, Any] = {}

    # Rosetta 指标 (带 Average_ 前缀)
    rosetta_map = {
        "Binder_Energy_Score": iface["binder_score"],
        "Surface_Hydrophobicity": iface["surface_hydrophobicity"],
        "ShapeComplementarity": iface["interface_sc"],
        "PackStat": iface["interface_packstat"],
        "dG": iface["interface_dG"],
        "dSASA": iface["interface_dSASA"],
        "dG/dSASA": iface["interface_dG_SASA_ratio"],
        "Interface_SASA_%": iface["interface_fraction"],
        "Interface_Hydrophobicity": iface["interface_hydrophobicity"],
        "n_InterfaceResidues": iface["interface_nres"],
        "n_InterfaceHbonds": iface["interface_interface_hbonds"],
        "InterfaceHbondsPercentage": iface["interface_hbond_percentage"],
        "n_InterfaceUnsatHbonds": iface["interface_delta_unsat_hbonds"],
        "InterfaceUnsatHbondsPercentage": iface["interface_delta_unsat_hbonds_percentage"],
    }
    for label, value in rosetta_map.items():
        data[f"Average_{label}"] = value

    # 二级结构
    for label, value in ss.items():
        data[f"Average_{label}"] = value

    # 冲突
    data["Average_Relaxed_Clashes"] = screening_result["clashes_after_relax"]
    data["Average_Unrelaxed_Clashes"] = screening_result["clashes_before_relax"]

    # DSSP 置信度
    data["Average_i_pLDDT"] = screening_result["i_pLDDT"]
    data["Average_ss_pLDDT"] = screening_result["ss_pLDDT"]

    # Interface AA
    data["Average_InterfaceAAs"] = screening_result["interface_AA"]

    # AF2/ESMFold2 置信度 (如果提供)
    if af2_confidence:
        for key, value in af2_confidence.items():
            data[f"Average_{key}"] = value

    return data


def screen_critic_results(
    critic_results: list[dict],
    dssp_path: str,
    filters_json: str | None = None,
    relax_first: bool = True,
    relaxed_output_dir: str | None = None,
    omit_AAs: str | None = "C",
) -> tuple[list[dict], list[dict]]:
    """对 ESMFold2 的 critic_results 执行 BindCraft 过滤。

    Parameters
    ----------
    critic_results : list[dict]
        ESMFold2 design_binder() 返回的第三个元素,
        每个 dict 包含:
          - "designed_sequence": "TARGETSEQ|BINDERSEQ"
          - "complex": ProteinComplex (Hero Critic 产出)
          - "critic_name": str
          - "is_scaling_critic": bool
          - "iptm": float | None
          - "distogram_iptm_proxy": float
          - ...
    dssp_path : str
        DSSP 可执行文件路径
    filters_json : str | None
        filter 阈值 JSON 路径 (None = 使用内置默认)
    relax_first : bool
        是否先 FastRelax
    relaxed_output_dir : str | None
        弛豫 PDB 保存目录
    omit_AAs : str | None
        禁用氨基酸, 默认 "C" (排除半胱氨酸)

    Returns
    -------
    passed : list[dict]
        通过所有过滤的设计
    failed : list[dict]
        未通过过滤的设计 (含未满足条件列表)
    """
    if filters_json is None:
        filters_json = os.path.join(
            os.path.dirname(__file__), "default_filters.json"
        )

    with open(filters_json, 'r') as f:
        filters = json.load(f)

    passed: list[dict] = []
    failed: list[dict] = []

    # 仅处理 Hero Critics (有 complex/全原子坐标)
    hero_results = [
        r for r in critic_results
        if not r.get("is_scaling_critic") and r.get("complex") is not None
    ]

    import tempfile

    for i, result in enumerate(hero_results):
        # 导出 PDB
        complex_obj = result["complex"]
        pdb_string = complex_obj.to_pdb_string()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False
        ) as tmp:
            tmp.write(pdb_string)
            pdb_path = tmp.name

        try:
            chains = [
                c for c in result["designed_sequence"].split("|") if c
            ]
            binder_seq = chains[-1]

            ##### claude 2026/8/13 start #####
            # 链名推导: target 链在前, binder 恒为最后一段。
            from esmcp.binder_design import _asym_id_to_chain_label

            labels = [_asym_id_to_chain_label(i) for i in range(len(chains))]
            target_labels = labels[:-1]
            binder_label = labels[-1]
            ##### claude 2026/8/13 end #####

            # ESMFold2 置信度
            af2_conf = {}
            if result.get("iptm") is not None:
                af2_conf["i_pTM"] = result["iptm"]
            if result.get("distogram_iptm_proxy") is not None:
                af2_conf["pLDDT"] = result["distogram_iptm_proxy"]
            if result.get("final_loss") is not None:
                af2_conf["pTM"] = max(0.0, 1.0 - result["final_loss"] / 10.0)

            ##### claude 2026/8/13 start #####
            screening = screen_single_pdb(
                pdb_path=pdb_path,
                binder_sequence=binder_seq,
                dssp_path=dssp_path,
                binder_chain=binder_label,
                target_chains=target_labels,
                relax_first=relax_first,
                relaxed_output_dir=relaxed_output_dir,
                omit_AAs=omit_AAs,
                af2_confidence=af2_conf,
            )
            ##### claude 2026/8/13 end #####

            data_dict = build_filter_data_dict(screening, af2_confidence=af2_conf)

            filter_result = check_filters(data_dict, filters)

            entry = {
                "designed_sequence": result["designed_sequence"],
                "critic_name": result.get("critic_name", "unknown"),
                "screening": screening,
                "data_dict": data_dict,
            }

            if filter_result is True:
                entry["filter_status"] = "PASSED"
                passed.append(entry)
            else:
                entry["filter_status"] = "FAILED"
                entry["unmet_conditions"] = filter_result
                failed.append(entry)

        finally:
            os.unlink(pdb_path)

    return passed, failed
