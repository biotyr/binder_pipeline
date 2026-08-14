"""
ProteinMPNN 逆折叠序列重设计模块。

对 ESMFold2 Hero Critic 输出的 PDB backbone 进行固定界面残基的
序列重采样, 类似 BindCraft 中 trajectory → MPNN 的流程。

核心逻辑:
1. 解析 PDB, 用 cKDTree 检测 binder 与 target 的界面残基
2. 固定 target 链全部残基 + binder 界面残基, 其余位置可突变
3. 调用 ColabDesign / ProteinMPNN 生成候选序列

依赖:
    - colabdesign.mpnn (mk_mpnn_model)
    - 本项目的 bindercraft_filter.structural_analysis (hotspot_residues)

使用方式::

    from inverse_optimize.mpnn_redesign import mpnn_redesign_binder

    sequences = mpnn_redesign_binder(
        pdb_path="/path/to/design.pdb",
        binder_chain="B",        # binder 链名必须显式提供 (无默认链假设)
        target_chains="A",       # 多链 target 时传 ["A", "B", ...]
        num_seqs=30,
        sampling_temp=0.1,
    )
    for entry in sequences:
        print(f"{entry['sequence']}  score={entry['score']:.2f}")
"""

from __future__ import annotations

import sys
from typing import Any

# ---- Type aliases ----

RedesignResult = dict[str, Any]
"""单条 MPNN 重设计结果::

    {
        "sequence": str,        # 单字母氨基酸序列
        "score": float,         # MPNN score (越低通常越好)
        "seqid": float,         # 与原始序列的序列一致性 (0-1)
        "length": int,          # 序列长度
    }
"""


def _import_hotspot_residues():
    """延迟导入以避免循环依赖。"""
    # 确保父包在 path 中
    import os as _os
    _pipeline_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pipeline_dir not in sys.path:
        sys.path.insert(0, _pipeline_dir)

    from bindercraft_filter.structural_analysis import hotspot_residues as _hs
    return _hs


def _detect_interface(
    pdb_path: str,
    binder_chain: str,
    target_chains: list[str],
    contact_cutoff: float = 4.0,
) -> dict[int, str]:
    """检测 binder 上与 target 链有原子接触的界面残基。

    Parameters
    ----------
    pdb_path : str
        复合体 PDB 文件。
    binder_chain : str
        Binder 链 ID。
    ##### claude 2026/8/13 start #####
    target_chains : list[str]
        Target 链 ID 列表 (多链 target 时为全部链)。
    ##### claude 2026/8/13 end #####
    contact_cutoff : float
        原子接触距离阈值 (Å)。

    Returns
    -------
    dict[int, str]
        {PDB_residue_number: one_letter_AA} 界面残基映射。
    """
    hotspot_residues = _import_hotspot_residues()
    ##### claude 2026/8/13 start #####
    # 多链 target: 传入全部 target 链, 检测 binder 与并集的界面接触
    return hotspot_residues(
        pdb_path,
        binder_chain,
        target_chains=target_chains,
        atom_distance_cutoff=contact_cutoff,
    )
    ##### claude 2026/8/13 end #####


def _build_fix_positions(
    interface_residues: dict[int, str],
    target_labels: list[str],
) -> str:
    ##### claude 2026/8/13 start #####
    """构建 MPNN fix_pos 字符串 (binder 为 design 第一条链)。

    colabdesign ``prep_pos`` 语义 (v1.1.1 源码核实):
    - 纯字母 token 固定整条链, 多链安全 (全部 target 链逐个列出)
    - 裸数字 token 应用于设计顺序的第一条链 (chain[0] = binder)

    因此 design_chains 把 binder 放在第一位 (见 :func:`_build_design_chains`),
    裸数字即 binder 的 PDB resSeq, 无需跨链 offset 计算。

    Parameters
    ----------
    interface_residues : dict[int, str]
        hotspot_residues() 的输出。
    target_labels : list[str]
        Target 链 ID 列表, 这些链上所有残基都会被固定。

    Returns
    -------
    str
        例如 ``"A,B,234,235,240"`` — 全部 target 链 ID 后跟
        逗号分隔的 binder 界面残基 PDB 编号 (裸数字作用于第一条
        链, 即 binder)。
    """
    binder_fixed = ",".join(str(r) for r in sorted(interface_residues.keys()))
    letters = ",".join(target_labels)
    if binder_fixed:
        return f"{letters},{binder_fixed}"
    return letters
    ##### claude 2026/8/13 end #####


def _build_design_chains(
    target_labels: list[str],
    binder_chain: str,
) -> str:
    ##### claude 2026/8/13 start #####
    """构建 MPNN design_chains 字符串 — binder 放在第一位。

    colabdesign ``prep_pdb`` 对第一条链不施加 residue offset, 使
    fix_pos 中作用于 chain[0] 的裸数字直接对应 binder 的 PDB resSeq。
    ProteinMPNN 采样每步随机打乱解码顺序, 链排列顺序不影响设计质量。

    Returns
    -------
    str
        例如 ``"C,A,B"`` (binder=C 在前, 两条 target 链在后)。
    """
    assert binder_chain not in target_labels
    return ",".join([binder_chain, *target_labels])
    ##### claude 2026/8/13 end #####


def mpnn_redesign_binder(
    pdb_path: str,
    binder_chain: str,
    target_chains: list[str] | str,
    num_seqs: int = 20,
    sampling_temp: float = 0.1,
    backbone_noise: float = 0.0,
    fix_interface: bool = True,
    interface_cutoff: float = 4.0,
    omit_AAs: str = "C",
    model_path: str = "v_48_020",
    mpnn_weights: str = "soluble",
    random_seed: int | None = None,
) -> list[RedesignResult]:
    """在固定 backbone 和界面残基的前提下, 用 ProteinMPNN 重设计 binder 序列。

    工作流:
    1. 检测界面残基 (cKDTree, 跨链原子距离 < interface_cutoff)
    2. 固定全部 target 链 + (可选) binder 界面残基
    3. MPNN 采样 ``num_seqs`` 条序列

    Parameters
    ----------
    pdb_path : str
        复合体 PDB 文件路径。
    binder_chain : str
        Binder 链 ID, 必须显式提供 (无默认链假设)。
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str
        Target 链 ID (单链 "A" 或多链 ["A", "B"])。
        design_chains 中 binder 放在第一位, 使 fix_pos 的裸数字
        直接作用于 binder (colabdesign prep_pos 语义)。
    ##### claude 2026/8/13 end #####
    num_seqs : int
        生成序列数量, 默认 20。
    sampling_temp : float
        MPNN 采样温度, 0.1 接近确定性, 1.0 高多样性。
    backbone_noise : float
        注入 backbone 高斯噪声的标准差 (Å), 默认 0.0。
        增大可提高序列多样性, 但可能降低结构兼容性。
    fix_interface : bool
        True = 固定 binder 界面残基不变 (类似 BindCraft 默认行为)。
        False = 允许 MPNN 重设计界面残基。
    interface_cutoff : float
        界面残基检测的原子距离阈值 (Å), 默认 4.0。
    omit_AAs : str
        逗号分隔的排除氨基酸单字母码, 默认 "C" (排除 Cys)。
    model_path : str
        ProteinMPNN checkpoint, 默认 "v_48_020"。
    mpnn_weights : str
        MPNN 权重后缀, 默认 "soluble" (可溶性偏向)。
        其他选项: "vanilla", "noise", 等。
    random_seed : int | None
        随机种子。None 表示不固定。

    Returns
    -------
    list[RedesignResult]
        按 MPNN score 升序排列的序列列表。每条记录包含:
        - sequence: str
        - score: float
        - seqid: float (与原始序列的序列一致性)
        - length: int
        - mpnn_n: int (序列编号, 1-indexed)

    Raises
    ------
    ImportError
        colabdesign 未安装。
    RuntimeError
        binder 链没有检测到界面残基。
    """
    from colabdesign.mpnn import mk_mpnn_model

    ##### claude 2026/8/13 start #####
    target_labels = [target_chains] if isinstance(target_chains, str) else list(target_chains)
    assert binder_chain not in target_labels, (
        f"binder_chain {binder_chain!r} must not appear in target_chains."
    )
    ##### claude 2026/8/13 end #####

    # ---- 1. 检测界面残基 ----
    interface_residues: dict[int, str] = {}
    if fix_interface:
        interface_residues = _detect_interface(
            pdb_path,
            binder_chain=binder_chain,
            target_chains=target_labels,
            contact_cutoff=interface_cutoff,
        )
        if not interface_residues:
            raise RuntimeError(
                f"No interface residues detected on chain {binder_chain} "
                f"at cutoff {interface_cutoff}Å. "
                f"Try increasing interface_cutoff or setting fix_interface=False."
            )

    # ---- 2. 构建 fix_pos 和 design_chains ----
    design_chains = _build_design_chains(target_labels, binder_chain)
    fix_positions = (
        _build_fix_positions(interface_residues, target_labels)
        if fix_interface
        else ",".join(target_labels)
    )

    # ---- 3. 初始化 MPNN ----
    mpnn_model = mk_mpnn_model(
        backbone_noise=backbone_noise,
        model_name=model_path,
        weights=mpnn_weights,
    )

    mpnn_model.prep_inputs(
        pdb_filename=pdb_path,
        chain=design_chains,
        fix_pos=fix_positions,
        rm_aa=omit_AAs,
    )

    # ---- 4. 采样序列 ----
    if random_seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    raw = mpnn_model.sample(
        temperature=sampling_temp,
        num=1,
        batch=num_seqs,
    )

    # ---- 5. 结构化输出 ----
    binder_len = len(raw["seq"][0]) if raw.get("seq") else 0
    # MPNN 返回的 seq 可能包含 target + binder; 取前 binder_len 个残基

    results: list[RedesignResult] = []
    for n in range(len(raw.get("seq", []))):
        full_seq = raw["seq"][n]
        ##### claude 2026/8/13 start #####
        # binder 在 design_chains 的第一位 (binder-first 设计),
        # 因此输出序列的前段即 binder (若返回全链拼接序列)。
        seq = full_seq[:binder_len] if binder_len and len(full_seq) > binder_len else full_seq
        ##### claude 2026/8/13 end #####

        results.append({
            "sequence": seq,
            "score": round(raw["score"][n], 4) if raw.get("score") else float("nan"),
            "seqid": round(raw["seqid"][n], 4) if raw.get("seqid") else float("nan"),
            "length": len(seq),
            "mpnn_n": n + 1,
        })

    # 按 score 升序 (更低 = 更好)
    results.sort(key=lambda r: r["score"] if not (r["score"] != r["score"]) else float("inf"))

    # ---- 6. 摘要 ----
    n_interface = len(interface_residues)
    _print_summary(pdb_path, len(results), n_interface, fix_positions)

    return results


def _print_summary(
    pdb_path: str,
    n_results: int,
    n_interface: int,
    fix_positions: str,
) -> None:
    """打印重设计摘要。"""
    print(f"[MPNN Redesign] {pdb_path}")
    print(f"  Generated: {n_results} sequences")
    if n_interface > 0:
        print(f"  Interface residues fixed: {n_interface}")
    else:
        print(f"  Interface: NOT fixed (full binder redesign)")
    print(f"  fix_pos: {fix_positions}")
    if n_results > 0:
        # best/worst scores already sorted
        print(f"  Score range: will be shown per-sequence")


def mpnn_redesign_from_critic_results(
    critic_results: list[dict[str, Any]],
    dssp_path: str = "",
    num_seqs: int = 20,
    sampling_temp: float = 0.1,
    fix_interface: bool = True,
    target_chains: list[str] | str | None = None,
    binder_chain: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """对 ESMFold2 critic_results 中所有 Hero Critic 输出的 PDB 执行
    MPNN 重设计。

    遍历 ``critic_results``, 筛选 ``is_scaling_critic=False`` 且
    ``complex is not None`` 的结果, 导出 PDB 后调用
    :func:`mpnn_redesign_binder`。

    Parameters
    ----------
    critic_results : list[dict]
        ESMFold2 ``design_binder()`` 的第三个返回值。
    dssp_path : str
        保留参数 (此函数不使用 DSSP, 但维持签名兼容)。
    num_seqs : int
        每个 PDB 生成的 MPNN 序列数。
    sampling_temp : float
        MPNN 采样温度。
    fix_interface : bool
        是否固定界面残基。
    ##### claude 2026/8/13 start #####
    target_chains : list[str] | str | None
        Target 链 ID (单链 "A" 或多链 ["A", "B"])。
        None 时从每条结果的 "designed_sequence" 自动推导。
    binder_chain : str | None
        Binder 链 ID。None 时从 "designed_sequence" 自动推导
        (复合物最后一条链的标签)。
    ##### claude 2026/8/13 end #####
    **kwargs
        传递给 ``mpnn_redesign_binder`` 的其他参数。

    Returns
    -------
    list[dict]
        每个元素包含:
        - "original_designed_sequence": str
        - "critic_name": str
        - "iptm": float|None
        - "mpnn_designs": list[RedesignResult]
    """
    import tempfile
    import os as _os

    ##### claude 2026/8/13 start #####
    def _labels(n_chains: int) -> list[str]:
        """按 asym_id 顺序推导链名 (0→A, 1→B, ...)。"""
        from esmcp.binder_design import _asym_id_to_chain_label

        return [_asym_id_to_chain_label(i) for i in range(n_chains)]
    ##### claude 2026/8/13 end #####

    hero_results = [
        r for r in critic_results
        if not r.get("is_scaling_critic") and r.get("complex") is not None
    ]

    if not hero_results:
        print("[MPNN Redesign] No Hero Critic results with complex found.")
        return []

    all_results: list[dict[str, Any]] = []

    for i, result in enumerate(hero_results):
        complex_obj = result["complex"]

        ##### claude 2026/8/13 start #####
        # 链名推导: 显式参数优先, 否则从 designed_sequence ("T1|T2|binder")
        # 自动推导 — target 链在前, binder 恒为最后一段。
        if target_chains is None or binder_chain is None:
            chains = [
                c for c in result.get("designed_sequence", "").split("|") if c
            ]
            labels = _labels(len(chains)) if chains else []
            if target_chains is None:
                target_chains_here: list[str] = labels[:-1] if labels else []
            else:
                target_chains_here = (
                    [target_chains]
                    if isinstance(target_chains, str)
                    else list(target_chains)
                )
            if binder_chain is None:
                binder_chain_here = labels[-1] if labels else ""
            else:
                binder_chain_here = binder_chain
        else:
            target_chains_here = (
                [target_chains]
                if isinstance(target_chains, str)
                else list(target_chains)
            )
            binder_chain_here = binder_chain
        ##### claude 2026/8/13 end #####

        # 优先使用 annotated_pdb (含 REMARK 质量指标)
        pdb_string = result.get("annotated_pdb")
        if pdb_string is None:
            pdb_string = complex_obj.to_pdb_string()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as tmp:
            tmp.write(pdb_string)
            pdb_path = tmp.name

        try:
            ##### claude 2026/8/13 start #####
            mpnn_designs = mpnn_redesign_binder(
                pdb_path=pdb_path,
                binder_chain=binder_chain_here,
                target_chains=target_chains_here,
                num_seqs=num_seqs,
                sampling_temp=sampling_temp,
                fix_interface=fix_interface,
                **kwargs,
            )
            ##### claude 2026/8/13 end #####
        except RuntimeError as exc:
            print(f"[MPNN Redesign] Skipping critic {result.get('critic_name', i)}: {exc}")
            mpnn_designs = []
        finally:
            _os.unlink(pdb_path)

        all_results.append({
            "original_designed_sequence": result.get("designed_sequence", ""),
            "critic_name": result.get("critic_name", "unknown"),
            "iptm": result.get("iptm"),
            "ptm": result.get("ptm"),
            "mpnn_designs": mpnn_designs,
        })

    total_designs = sum(len(r["mpnn_designs"]) for r in all_results)
    print(
        f"[MPNN Redesign] Processed {len(hero_results)} PDBs → "
        f"{total_designs} total MPNN sequences"
    )
    return all_results
