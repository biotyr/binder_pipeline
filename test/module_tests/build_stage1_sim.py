#!/usr/bin/env python3
"""
Stage 1 模拟输出构建器 — 用真实晶体结构伪造 critic_results 的 annotated_pdb。

单链场景 (3RRQ 任务):
    target = PD-1  (4ZQK author chain B -> 新链 A)
    伪 binder = PD-L1 (4ZQK author chain A -> 新链 B, 重排为末链)
多链场景 (4ZQK 任务):
    target = PD-L1 (chain A) + PD-1 (chain B)  [双链 target]
    伪 binder = 3RRQ PD-1 叠合到 4ZQK chain B 后刚性旋转摆位 -> 新链 C

产物:
    sim_pdbs/  单链/多链 × pass/边界/冲突 样本 PDB
    sim_manifest.json  每个样本的 designed_sequence / 链角色 / 受控指标

真实晶体结构 (剥 HETATM) + pipeline 自己的 pdb_with_quality_remarks 注入
受控 REMARK 250 + 受控 B-factor pLDDT。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, PDBIO, Select

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from esmcp.binder_design import pdb_with_quality_remarks  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parent / "sim_pdbs"
OUT.mkdir(exist_ok=True)

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
CA_CLASH_CUTOFF = 2.5
CONTACT_CUTOFF = 4.0


# ---- 原子级读取: 剥 HETATM/水, 取标准残基 ATOM 行 ----

def load_chain_lines(pdb_path: str, chain_id: str) -> list[str]:
    """返回指定链的标准氨基酸 ATOM 行 (保留 altloc=' ' 或首个 altloc)。"""
    lines: list[str] = []
    seen: set[tuple[int, str]] = set()
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain_id:
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if line[17:20].strip() not in THREE2ONE:
            continue
        altloc = line[16]
        key = (resnum, line[12:16])
        if altloc not in (" ", "A"):
            continue
        if key in seen:
            continue
        seen.add(key)
        lines.append(line.rstrip("\n"))
    if not lines:
        raise ValueError(f"{pdb_path} chain {chain_id}: no standard-ATOM lines")
    return lines


def seq_of(lines: list[str]) -> str:
    seq, last = "", None
    for ln in lines:
        resnum = int(ln[22:26])
        if resnum != last:
            seq += THREE2ONE[ln[17:20].strip()]
            last = resnum
    return seq


def resnums_of(lines: list[str]) -> list[int]:
    out, last = [], None
    for ln in lines:
        r = int(ln[22:26])
        if r != last:
            out.append(r)
            last = r
    return out


def coords_of(lines: list[str], ca_only: bool = False) -> np.ndarray:
    out = []
    for ln in lines:
        if ca_only and ln[12:16].strip() != "CA":
            continue
        out.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.array(out)


# ---- 度量: Cα 冲突 / 重原子最小距离 / 接触残基数 ----

def ca_res_info(lines: list[str]) -> tuple[np.ndarray, list[tuple[str, int]]]:
    coords, info = [], []
    for ln in lines:
        if ln[12:16].strip() != "CA":
            continue
        coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        info.append((ln[21], int(ln[22:26])))
    return np.array(coords), info


def cross_chain_ca_clashes(lines_a: list[str], lines_b: list[str]) -> int:
    """两组链行之间 Cα-2.5Å 冲突对数 (跨链, 排除同残基)。"""
    ca_a, info_a = ca_res_info(lines_a)
    ca_b, info_b = ca_res_info(lines_b)
    if len(ca_a) < 1 or len(ca_b) < 1:
        return 0
    tree = cKDTree(ca_a)
    pairs = tree.query_ball_tree(cKDTree(ca_b), CA_CLASH_CUTOFF)
    n = 0
    for i, js in enumerate(pairs):
        for j in js:
            n += 1  # 不同链组之间无同残基/相邻残基问题
    return n


def min_heavy_dist(lines_a: list[str], lines_b: list[str],
                   target_tree: cKDTree | None = None) -> float:
    tree = target_tree if target_tree is not None else cKDTree(coords_of(lines_a))
    d, _ = tree.query(coords_of(lines_b), k=1)
    return float(d.min())


def contact_residues(binder_lines: list[str], target_lines: list[str],
                     target_tree: cKDTree | None = None) -> set[int]:
    """binder 上与 target 有 ≤4Å 原子接触的残基号集合。"""
    tree = target_tree if target_tree is not None else cKDTree(coords_of(target_lines))
    hits = cKDTree(coords_of(binder_lines)).query_ball_tree(tree, CONTACT_CUTOFF)
    res = set()
    for ln, close in zip(binder_lines, hits):
        if close:
            res.add(int(ln[22:26]))
    return res


# ---- 刚性变换 ----

def rot_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rigid_move(lines: list[str], R: np.ndarray, t: np.ndarray) -> list[str]:
    out = []
    for ln in lines:
        c = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        c2 = R @ c + t
        out.append(
            ln[:30] + f"{c2[0]:8.3f}{c2[1]:8.3f}{c2[2]:8.3f}" + ln[54:]
        )
    return out


# ---- PDB 组装: 重编链名 + B-factor pLDDT + REMARK ----

def relabel(lines: list[str], new_chain: str) -> list[str]:
    return [ln[:21] + new_chain + ln[22:] for ln in lines]


def set_bfactor(lines: list[str], plddt_frac: float) -> list[str]:
    b = f"{plddt_frac * 100.0:6.2f}"
    return [ln[:60] + b + ln[66:] for ln in lines]


def assemble(chain_blocks: dict[str, list[str]], *, iptm: float, ptm: float,
             proxy: float, binder_plddt: float, target_plddt: float = 0.90) -> str:
    parts = []
    for chain_id, lines in chain_blocks.items():
        plddt = binder_plddt if chain_id == max(chain_blocks) else target_plddt
        # binder = 字典序最后一个链 (A<B<C) — 单链场景 binder=B, 多链场景 binder=C
        parts.append(relabel(set_bfactor(lines, plddt), chain_id))
        parts.append([f"TER   {len(parts):>3}"])
    body = []
    for block in parts:
        body.extend(block)
    body.append("END")
    pdb = "\n".join(body) + "\n"
    return pdb_with_quality_remarks(
        pdb, iptm=iptm, ptm=ptm, distogram_binding_confidence=proxy
    )


# ---- 主流程 ----

def main() -> None:
    parser = PDBParser(QUIET=True)
    manifest: dict[str, dict] = {}

    # ============ 单链场景: 4ZQK 链重排 ============
    pdb4 = str(DATA / "4ZQK.pdb")
    pd1_lines = load_chain_lines(pdb4, "B")     # PD-1  author B, res 33-146
    pdl1_lines = load_chain_lines(pdb4, "A")    # PD-L1 author A, res 18-132

    # 真实晶体界面自检: 无 Cα 冲突, 有接触
    assert cross_chain_ca_clashes(pd1_lines, pdl1_lines) == 0, \
        "4ZQK 真实界面出现 Cα 冲突, 构建假设失效"
    n_contact = len(contact_residues(pdl1_lines, pd1_lines))
    assert n_contact >= 10, f"PD-L1 伪 binder 界面残基过少: {n_contact}"
    print(f"[single] PD-1 target {len(pd1_lines)} atoms / {seq_of(pd1_lines)[:20]}..., "
          f"PD-L1 pseudo-binder interface residues = {n_contact}")

    single_blocks = {"A": pd1_lines, "B": pdl1_lines}  # binder = 末链 B
    single_variants = {
        "pass":        dict(iptm=0.62, ptm=0.55, proxy=0.58, binder_plddt=0.85),
        "fail_iptm":   dict(iptm=0.49, ptm=0.55, proxy=0.58, binder_plddt=0.85),
        "fail_plddt":  dict(iptm=0.62, ptm=0.55, proxy=0.58, binder_plddt=0.60),
        "fail_proxy":  dict(iptm=0.62, ptm=0.55, proxy=0.45, binder_plddt=0.85),
    }
    for name, kw in single_variants.items():
        fname = f"single_{name}.pdb"
        (OUT / fname).write_text(assemble(single_blocks, **kw))
        manifest[fname] = {
            "scenario": "single", "variant": name,
            "target_chains": ["A"], "binder_chain": "B",
            "designed_sequence": f"{seq_of(pd1_lines)}|{seq_of(pdl1_lines)}",
            "binder_sequence": seq_of(pdl1_lines),
            **kw,
        }

    # ============ 多链场景: 3RRQ PD-1 叠合 + 旋转摆位 ============
    pdb3 = str(DATA / "3RRQ.pdb")
    bnd_lines = load_chain_lines(pdb3, "A")     # 3RRQ PD-1, res 33-149

    # 按相同 author 残基号匹配 Cα 做 Kabsch 叠合 (同为 PD-1 UniProt 编号)
    def ca_line_map(lines: list[str]) -> dict[int, str]:
        return {int(l[22:26]): l for l in lines if l[12:16].strip() == "CA"}

    b4, b3 = ca_line_map(pd1_lines), ca_line_map(bnd_lines)
    common = sorted(set(b4) & set(b3))
    P = coords_of([b3[r] for r in common])
    Q = coords_of([b4[r] for r in common])
    assert len(common) >= 90, f"叠合匹配残基过少: {len(common)}"
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R0 = Vt.T @ np.diag([1, 1, d]) @ U.T
    t0 = Q.mean(0) - R0 @ P.mean(0)
    bnd_lines = rigid_move(bnd_lines, R0, t0)
    # 叠合自检: 与 4ZQK chain B 的 Cα RMSD 应很小
    rmsd = float(np.sqrt(((P @ R0.T + t0 - Q) ** 2).sum(1).mean()))
    print(f"[multi] superpose 3RRQ PD-1 onto 4ZQK chain B: "
          f"{len(common)} CA matched, RMSD={rmsd:.2f} Å")
    assert rmsd < 2.5, f"叠合 RMSD 异常: {rmsd}"  # 游离/结合态 PD-1 允许 ~1-2 Å 构象差

    # 冲突样本: 零旋转 (与 chain B 重合)
    clash_blocks = {"A": pdl1_lines, "B": pd1_lines, "C": bnd_lines}
    fname = "multi_clash.pdb"
    (OUT / fname).write_text(assemble(
        clash_blocks, iptm=0.75, ptm=0.60, proxy=0.58, binder_plddt=0.85))
    manifest[fname] = {
        "scenario": "multi", "variant": "clash",
        "target_chains": ["A", "B"], "binder_chain": "C",
        "designed_sequence": f"{seq_of(pdl1_lines)}|{seq_of(pd1_lines)}|{seq_of(bnd_lines)}",
        "binder_sequence": seq_of(bnd_lines),
        "iptm": 0.75, "ptm": 0.60, "proxy": 0.58, "binder_plddt": 0.85,
    }

    # 摆位搜索: 伪 binder 先绕若干轴预旋转 (平面朝向 target), 再从复合物
    # 质心沿确定性方向平移, 找 无 Cα 冲突 + 重原子 3.0-4.0Å 接触 + ≥8 接触残基
    target_all = pdl1_lines + pd1_lines
    t_centroid = coords_of(target_all).mean(0)
    # ---- 数组化搜索 (避免逐 pose 字符串替换/建树) ----
    b_heavy0 = coords_of(bnd_lines)
    b_ca0 = coords_of([l for l in bnd_lines if l[12:16].strip() == "CA"])
    atom_res = np.array([int(l[22:26]) for l in bnd_lines])
    target_tree = cKDTree(coords_of(target_all))
    t_ca_tree = cKDTree(coords_of(
        [l for l in target_all if l[12:16].strip() == "CA"]))

    dirs = [np.array(v, float) / np.linalg.norm(v) for v in
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
             (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
             (1, 0, 1), (0, 1, 1), (1, 1, 1), (-1, 1, 1)]]
    ori_axes = [np.array(v, float) / np.linalg.norm(v) for v in
                [(1, 1, 0), (1, -1, 0), (0, 1, 0), (1, 0, 0)]]
    placed, chosen, win_t = None, None, None
    for ax, deg in [(a, d) for a in ori_axes for d in (30.0, 60.0)]:
        R = rot_matrix(ax, np.radians(deg))
        rc = b_heavy0.mean(0)
        rot_heavy = (R @ (b_heavy0 - rc).T).T + rc
        rot_ca = (R @ (b_ca0 - rc).T).T + rc
        for direction in dirs:
            for sep in np.arange(20.0, 60.01, 0.25):
                t = t_centroid + direction * sep - rc
                # 重原子窗口 (2.5, 4.0]: 允许天然界面常见的 2.6-3.0Å 侧链
                # 近接触; Cα 冲突 (<2.5Å) 由下面单独把关
                dmin = float(target_tree.query(rot_heavy + t)[0].min())
                if not (CA_CLASH_CUTOFF < dmin <= CONTACT_CUTOFF):
                    continue
                if (t_ca_tree.query(rot_ca + t)[0] < CA_CLASH_CUTOFF).any():
                    continue
                pairs = cKDTree(rot_heavy + t).query_ball_tree(
                    target_tree, CONTACT_CUTOFF)
                n_contact = len({atom_res[i] for i, js in enumerate(pairs) if js})
                if n_contact < 8:
                    continue
                placed, chosen = True, (
                    direction.round(3).tolist(), float(deg), float(sep), dmin,
                    n_contact)
                win_R, win_rc, win_t = R, rc, t
                break
            if placed:
                break
        if placed:
            break
    assert placed is not None, "摆位搜索失败: 未找到无冲突且接触的摆位"
    # 胜出摆位一次性回写 PDB 行: 先绕 rc 旋转 R, 再平移 win_t
    binder_moved = rigid_move(bnd_lines, win_R, win_rc - win_R @ win_rc + win_t)
    print(f"[multi] placement: dir={chosen[0]} deg={chosen[1]} sep={chosen[2]} "
          f"min_dist={chosen[3]:.2f} Å contacts={chosen[4]}")

    multi_blocks = {"A": pdl1_lines, "B": pd1_lines, "C": binder_moved}
    multi_variants = {
        "pass":      dict(iptm=0.75, ptm=0.60, proxy=0.58, binder_plddt=0.85),
        "fail_iptm": dict(iptm=0.69, ptm=0.60, proxy=0.58, binder_plddt=0.85),
    }
    for name, kw in multi_variants.items():
        fname = f"multi_{name}.pdb"
        (OUT / fname).write_text(assemble(multi_blocks, **kw))
        manifest[fname] = {
            "scenario": "multi", "variant": name,
            "target_chains": ["A", "B"], "binder_chain": "C",
            "designed_sequence": (
                f"{seq_of(pdl1_lines)}|{seq_of(pd1_lines)}|{seq_of(binder_moved)}"),
            "binder_sequence": seq_of(binder_moved),
            "placement": {"dir": chosen[0], "deg": chosen[1], "sep": chosen[2]},
            **kw,
        }

    (Path(__file__).parent / "sim_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\n[done] {len(manifest)} simulated PDBs -> {OUT}")
    for f in manifest:
        print("  ", f)


if __name__ == "__main__":
    main()
