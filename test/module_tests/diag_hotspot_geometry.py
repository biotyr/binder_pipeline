#!/usr/bin/env python3
"""
热点局部几何验证: 交界边缘贴片"卡得住" vs 单链凸面贴片"滑落"假说。

对 4ZQK 四个热点 (B:S30/A:E28 = 臂i 对, A:Q30/A:D73 = 臂ii 对) 从晶体结构
计算每残基局部形貌, 检验假说:
- 臂i 的 B:S30 位于 A/B 界面凹槽缘 → 有对侧链"墙" (跨链邻居), 局部方向性强
  (邻居聚在一侧 = 有依托), 搭附稳定 → P1 锚定快 (epi 1.49@25);
- 臂ii 的 Q30/D73 位于链 A 凸面/开阔面 → 无对侧墙, 邻居各向同性 (无依托),
  搭附滑落 → P1 锚定慢 (2.65) 且全程翻转震荡。

指标 (全部由 Cβ/CA 坐标计算):
- RSA: mkdssp 复合体溶剂可及性 (部分埋藏 = 界面 rim 特征)
- cross8/cross10: 对侧链 Cβ 邻居数 (≤8/10Å) — 界面边缘的直接证据
- prot: 凸出度 = (Cβ-局部形心)·法向 / 切向 RMS 半径 (PCA 局部平面拟合;
  高=凸钮, ~0=平, 负=凹陷)
- wall: 切向邻居方向合力长度 [0,1] (≈0=各向同性凸钮无墙; >0.5=一侧有墙/槽缘)
- same10: 同链 Cβ 邻居数 (≤10Å, 序列间隔≥3) — 局部表面密度

用法: python diag_hotspot_geometry.py  (无 GPU 依赖)
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

_pipeline_dir = Path(__file__).resolve().parents[2]
PDB_FILE = _pipeline_dir / "test" / "data" / "4ZQK.pdb"
DSSP_BIN = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"
OUT_DIR = _pipeline_dir / "test" / "module_results" / "diag_hotspot_geometry"

# 热点: seq 链内位置 → author 编号 (A 偏移 17, B 偏移 32, 先前会话已验证)
HOTSPOTS = {
    "B:S30": ("B", 62, "S"),
    "A:E28": ("A", 45, "E"),
    "A:Q30": ("A", 47, "Q"),
    "A:D73": ("A", 90, "D"),
}
MAX_ASA = {"ALA": 129, "ARG": 248, "ASN": 157, "ASP": 163, "CYS": 167,
           "GLN": 198, "GLU": 194, "GLY": 146, "HIS": 184, "ILE": 183,
           "LEU": 180, "LYS": 205, "MET": 201, "PHE": 222, "PRO": 159,
           "SER": 155, "THR": 172, "TRP": 264, "TYR": 236, "VAL": 170}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def parse_pdb() -> dict:
    """author 链 → {author_num: {aa, cb, ca}}"""
    chains: dict[str, dict[int, dict]] = {}
    for ln in PDB_FILE.read_text().split("\n"):
        if not ln.startswith("ATOM") or ln[12:16].strip() not in ("CA", "CB"):
            continue
        if ln[16] not in (" ", "A"):
            continue
        num, aa3 = int(ln[22:26]), ln[17:20].strip()
        if aa3 not in THREE2ONE:
            continue
        ch = chains.setdefault(ln[21], {})
        rec = ch.setdefault(num, {"aa": THREE2ONE[aa3], "cb": None, "ca": None,
                                  "num": num})
        rec["cb" if ln[12:16].strip() == "CB" else "ca"] = (
            float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return chains


def run_dssp_rsa() -> dict:
    txt = subprocess.run([DSSP_BIN, "-i", str(PDB_FILE)],
                         capture_output=True, text=True, timeout=120,
                         check=True).stdout
    rsa = {}
    for ln in txt.split("\n"):
        if len(ln) < 39 or ln[13] == "!":
            continue
        try:
            auth, num = ln[11], int(ln[5:10])
            aa, acc = ln[13], int(ln[34:38].strip() or "-1")
        except ValueError:
            continue
        if auth in ("!", "") or aa not in THREE2ONE.values() or acc < 0:
            continue
        three = [k for k, v in THREE2ONE.items() if v == aa][0]
        rsa[(auth, num)] = acc / MAX_ASA[three]
    return rsa


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def pca_normal(pts):
    """邻居点集的最小特征向量 (局部平面法向), 雅可比幂迭代即可。"""
    c = scale(sum(pts, start=(0.0, 0.0, 0.0)), 1.0 / len(pts))
    q = [sub(p, c) for p in pts]
    # 协方差矩阵
    cov = [[sum(p[i] * p[j] for p in q) for j in range(3)] for i in range(3)]
    # 幂迭代求最大特征向量 → 法向为最小, 用最大特征向量正交化空间里的最小方向:
    # 3x3 小矩阵直接求特征向量 (解析或迭代): 用反幂迭代在正交补里找最小。
    v = (1.0, 0.0, 0.0)
    for _ in range(100):  # 幂迭代 → 最大
        w = (dot(cov[0], v), dot(cov[1], v), dot(cov[2], v))
        n = norm(w)
        if n < 1e-12:
            break
        v = scale(w, 1.0 / n)
    # 正交补内的反幂迭代求最小: 对 (I - vv^T) A (I - vv^T) 做幂迭代等价于
    # 迭代 w = A v2 后从 v2 中扣除 v 分量再归一, 但反幂需逆。
    # 简化: 3x3 特征分解 via Characteristic polynomial 的 Hardy 迭代不划算 —
    # 用三次正交迭代 (Jacobi) 的紧凑实现:
    import numpy as np
    w, vecs = np.linalg.eigh(np.array(cov))
    normal = vecs[:, 0]  # 最小特征值对应列
    return c, (float(normal[0]), float(normal[1]), float(normal[2]))


def local_topo(cb, own_neighbors, global_down):
    """凸出度 prot 与墙方向性 wall。

    own_neighbors: 同链 Cβ 列表 (已按半径与序列间隔过滤)
    global_down: "指向蛋白内部"的参考方向 (链全局形心 → 该残基的反向)
    """
    if len(own_neighbors) < 5:
        return float("nan"), float("nan"), len(own_neighbors)
    centroid, normal = pca_normal(own_neighbors)
    if dot(normal, global_down) < 0:  # 法向朝外
        normal = scale(normal, -1.0)
    h = dot(sub(cb, centroid), normal)
    tang_rms = math.sqrt(sum(dot(sub(n, centroid), sub(n, centroid))
                             for n in own_neighbors) / len(own_neighbors)
                         - max(0.0, h) ** 2)
    prot = h / max(tang_rms, 1e-6)
    # 墙方向性: 邻居相对残基的切向单位向量合力
    dirs = []
    for n in own_neighbors:
        d = sub(n, cb)
        d_t = sub(d, scale(normal, dot(d, normal)))
        L = norm(d_t)
        if L > 1e-6:
            dirs.append(scale(d_t, 1.0 / L))
    if not dirs:
        return prot, float("nan"), len(own_neighbors)
    m = scale(sum(dirs, start=(0.0, 0.0, 0.0)), 1.0 / len(dirs))
    wall = norm(m)
    return prot, wall, len(own_neighbors)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chains = parse_pdb()
    rsa = run_dssp_rsa()

    # 链全局形心 (法向朝向参考)
    def chain_centroid(cl):
        pts = [r["cb"] or r["ca"] for r in chains[cl].values()
               if r["cb"] or r["ca"]]
        return scale(sum(pts, start=(0.0, 0.0, 0.0)), 1.0 / len(pts))

    centroids = {cl: chain_centroid(cl) for cl in chains}

    def coord(cl, num):
        rec = chains[cl][num]
        return rec["cb"] or rec["ca"]

    print(f"{'热点':>7} {'aa':>3} {'RSA':>5} {'cross8':>6} {'cross10':>7} "
          f"{'same12':>6} {'prot':>6} {'wall':>5}", flush=True)
    rows = {}
    for hs, (cl, num, letter) in HOTSPOTS.items():
        rec = chains[cl].get(num)
        assert rec is not None and rec["aa"] == letter, (hs, rec)
        cb = rec["cb"] or rec["ca"]
        r = rsa.get((cl, num), float("nan"))
        cross8 = cross10 = 0
        for other in chains:
            if other == cl:
                continue
            for o in chains[other].values():
                oc = o["cb"] or o["ca"]
                if not oc:
                    continue
                d = norm(sub(oc, cb))
                cross8 += d < 8
                cross10 += d < 10
        own = [o["cb"] or o["ca"] for k, o in chains[cl].items()
               if abs(k - num) >= 2 and (o["cb"] or o["ca"])
               and norm(sub(o["cb"] or o["ca"], cb)) < 12.0]
        gdown = scale(sub(cb, centroids[cl]), -1.0)  # 形心→残基 = 朝外
        prot, wall, n = local_topo(cb, own, gdown)
        rows[hs] = {"rsa": r, "cross8": cross8, "cross10": cross10,
                    "same12": n, "prot": round(prot, 2), "wall": round(wall, 2)}
        print(f"{hs:>7} {letter:>3} {r:>5.2f} {cross8:>6} {cross10:>7} "
              f"{n:>6} {prot:>6.2f} {wall:>5.2f}", flush=True)

    # 背景对照: 两链全部可及表面残基 (RSA>=0.4) 的 prot/wall 分布
    import statistics as st
    bg = {"A": [], "B": []}
    bg_wall = {"A": [], "B": []}
    for cl in ("A", "B"):
        for k, o in chains[cl].items():
            r = rsa.get((cl, k), float("nan"))
            oc = o["cb"] or o["ca"]
            if not (r >= 0.4 and oc):
                continue
            own_bg = [x["cb"] or x["ca"] for kk, x in chains[cl].items()
                      if abs(kk - k) >= 2 and (x["cb"] or x["ca"])
                      and norm(sub(x["cb"] or x["ca"], oc)) < 12.0]
            gd = scale(sub(oc, centroids[cl]), -1.0)
            pr, wl, _ = local_topo(oc, own_bg, gd)
            if pr == pr:
                bg[cl].append(pr)
            if wl == wl:
                bg_wall[cl].append(wl)
    print("\n背景分布 (RSA>=0.4 表面残基):", flush=True)
    for cl in ("A", "B"):
        v, w = bg[cl], bg_wall[cl]
        if v:
            print(f"  链{cl}: prot 中位 {st.median(v):.2f} "
                  f"[IQR {st.quantiles(v, n=4)[0]:.2f}-{st.quantiles(v, n=4)[2]:.2f}, n={len(v)}] | "
                  f"wall 中位 {st.median(w):.2f} [IQR {st.quantiles(w, n=4)[0]:.2f}-{st.quantiles(w, n=4)[2]:.2f}]",
                  flush=True)

    print("\n热点间距 (Å, Cβ):", flush=True)
    dists = {}
    keys = list(HOTSPOTS)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = norm(sub(coord(HOTSPOTS[a][0], HOTSPOTS[a][1]),
                         coord(HOTSPOTS[b][0], HOTSPOTS[b][1])))
            dists[f"{a}-{b}"] = round(d, 1)
            print(f"  {a} - {b}: {d:.1f}", flush=True)

    (OUT_DIR / "result.json").write_text(json.dumps(
        {"hotspots": rows, "distances": dists}, indent=2))
    print(f"\n产物 → {OUT_DIR / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
