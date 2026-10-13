#!/usr/bin/env python3
"""
诊断: 在 4ZQK 上提取热点可及邻域 mask, 并用晶体结构做金标准校验。

平行模式 inter_loss_mode="hotspot_neighborhood" 的阈值校准实验:
- 生产路径: bd.precompute_inter_neighborhood_mask 在默认阈值下提 mask;
- 阈值扫描: R ∈ {10,12,15,18,20} × burial_quantile ∈ {0.3,0.5,0.6,0.7},
  与"晶体结构同公式真值 mask"逐组合对比 Jaccard;
- 金标准 (4ZQK.pdb 晶体结构, 复合体语境):
  * 真实几何邻域: Cβ(gly 用 CA) 距任一热点 Cβ < R;
  * 真实接触数: Cβ 距离 < cutoff 计数 (排除扁平序列 |i-j|<sep, 与代理同口径);
  * DSSP RSA: mkdssp 求复合体溶剂可及性, RSA=ACC/maxASA(Tien 2013),
    埋藏<20% / 部分 20-40% / 暴露>40%。
- 关键问题: (a) distogram 代理接触数能否分辨埋藏 (Spearman vs 真接触数/RSA);
  (b) 邻域 mask 是否排除界面埋藏核心、保留外表面 (纯度 = RSA≥20% 占比);
  (c) B:S30+A:E28 (交界边缘对) 的邻域构成。

用法: CUDA_VISIBLE_DEVICES=<GPU-UUID> python diag_inter_mask_4zqk.py
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
for p in (str(_pipeline_dir), str(_pipeline_dir / "test" / "e2e")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Bio import SeqIO  # noqa: E402
import torch  # noqa: E402

import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import ESMFold2Design  # noqa: E402

# Patch A (照搬 test/e2e/run_e2e.py, 本脚本不导入 run_e2e 故需自带):
# Transition._addmm_residual 位于 autocast 区块之外的 confidence-head 链
# (experimental:1079 → common:2716 → 2673), bf16 激活 × fp32 权重在此裸撞
# (compile 期 inductor pad_mm 基准测试与 eager 均崩)。包一层 autocast bf16。
from transformers.models.esmfold2 import modeling_esmfold2_common as _common

_orig_addmm_residual = _common.Transition._addmm_residual

def _addmm_residual_ac(self, x, hidden):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _orig_addmm_residual(self, x, hidden)

_common.Transition._addmm_residual = _addmm_residual_ac

TASK = "4ZQK"
BINDER_LEN = 94
SEED = 0
DSSP_BIN = "/home/taoyr/anaconda3/envs/esmbind/bin/dssp"
# fasta/complex 链序 → PDB author 链与编号偏移 (author = OFFSET + seq 1-based idx)
# 先前会话已验证: B:S62→seq30, A:S117→seq100。
PDB_FILE = _pipeline_dir / "test" / "data" / "4ZQK.pdb"
FASTA_FILE = _pipeline_dir / "test" / "data" / "rcsb_pdb_4ZQK.fasta"
HOTSPOT_FILE = _pipeline_dir / "test" / "data" / "hotspot.txt"
OUT_DIR = _pipeline_dir / "test" / "module_results" / "diag_inter_mask_4zqk"

# Tien et al. 2013 理论最大 ASA (Å²)
MAX_ASA = {"ALA": 129, "ARG": 248, "ASN": 157, "ASP": 163, "CYS": 167,
           "GLN": 198, "GLU": 194, "GLY": 146, "HIS": 184, "ILE": 183,
           "LEU": 180, "LYS": 205, "MET": 201, "PHE": 222, "PRO": 159,
           "SER": 155, "THR": 172, "TRP": 264, "TYR": 236, "VAL": 170}
THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
R_SWEEP = [10.0, 12.0, 15.0, 18.0, 20.0]
Q_SWEEP = [0.3, 0.5, 0.6, 0.7]


def load_hotspots() -> list[str]:
    lines = HOTSPOT_FILE.read_text().split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() == TASK and i + 1 < len(lines):
            return [h for h in re.split(r"\s+", lines[i + 1].strip()) if h]
    raise RuntimeError(f"hotspot.txt 中未找到 {TASK}")


def parse_pdb_residues() -> dict:
    """author 链 → {(author_num): (aa1, cb_coord)}; cb 缺失用 ca。"""
    chains: dict[str, dict[int, dict]] = {}
    for ln in PDB_FILE.read_text().split("\n"):
        if not ln.startswith("ATOM") or ln[12:16].strip() not in ("CA", "CB"):
            continue
        if ln[16] not in (" ", "A"):  # altloc
            continue
        auth_num = int(ln[22:26])
        aa3 = ln[17:20].strip()
        if aa3 not in THREE2ONE:
            continue
        ch = chains.setdefault(ln[21], {})
        rec = ch.setdefault(auth_num, {"aa": THREE2ONE[aa3], "cb": None, "ca": None})
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        key = "cb" if ln[12:16].strip() == "CB" else "ca"
        rec[key] = xyz
    return chains


def run_dssp_rsa() -> dict[tuple[str, int], float]:
    """mkdssp 2.0.4 经典输出 → {(author_chain, author_num): RSA}。"""
    txt = subprocess.run(
        [DSSP_BIN, "-i", str(PDB_FILE)], capture_output=True, text=True,
        timeout=120, check=True,
    ).stdout
    rsa = {}
    for ln in txt.split("\n"):
        if len(ln) < 39 or ln[13] == "!":
            continue
        try:
            auth = ln[11]
            num = int(ln[5:10])
            aa = ln[13]
            acc = int(ln[34:38].strip() or "-1")
        except ValueError:
            continue
        if auth in ("!", "") or aa not in THREE2ONE.values() or acc < 0:
            continue
        # 经典输出 aa 列是 1 字母; 对应回 3 字母查 maxASA
        three = [k for k, v in THREE2ONE.items() if v == aa][0]
        rsa[(auth, num)] = acc / MAX_ASA[three]
    return rsa


def spearman(a, b) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        ra = {v: r for r, v in enumerate(sorted(a))}
        rb = {v: r for r, v in enumerate(sorted(b))}
        # 平均秩修正略去 (诊断用足够)
        import numpy as np
        return float(np.corrcoef([ra[v] for v in a], [rb[v] for v in b])[0, 1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hotspots = load_hotspots()
    print(f"[mask4zqk] 热点 = {hotspots}", flush=True)

    seqs = {}
    for r in SeqIO.parse(FASTA_FILE, "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        seqs[m.group(1)] = str(r.seq).replace("X", "")
    chain_labels = ["A", "B"]  # complex asym 顺序 = fasta 链序
    target_seq = f"{seqs['A']}|{seqs['B']}"
    total_target = len(target_seq.replace("|", ""))
    print(f"[mask4zqk] target: A(PD-L1)={len(seqs['A'])}aa + B(PD-1)={len(seqs['B'])}aa",
          flush=True)

    # ── 生产路径提 mask ──────────────────────────────────────────
    app = ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    model = next(iter(app.inversion_models.values()))
    target_one_hot = bd.build_target_one_hot(target_seq, device="cuda")
    print(f"[mask4zqk] 模型加载 {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    mask_default = bd.precompute_inter_neighborhood_mask(
        model, target_seq, target_one_hot, BINDER_LEN, hotspots,
        batch_size=1, device="cuda", seed=SEED,
    )
    print(f"[mask4zqk] 生产路径预计算 {time.time()-t0:.0f}s → "
          f"{'None' if mask_default is None else int(mask_default.sum())} 残基",
          flush=True)
    assert mask_default is not None, "默认阈值下邻域过小, 需先放宽 min_size 再校准"

    # ── 自主折叠取原始量 (与生产路径同一口径) ─────────────────────
    dummy = torch_zeros_polya(1, BINDER_LEN)
    pre = bd.fold_and_get_distogram(
        model, target_seq, target_one_hot, dummy,
        num_loops=1, num_sampling_steps=1, calculate_confidence=False, seed=SEED,
    )
    dgram = pre["distogram_logits"][:, :total_target, :total_target].float()
    bins = bd.get_mid_points().to(dgram.device)
    probs = torch.softmax(dgram, dim=-1)
    exp_pair = (probs * bins).sum(-1)[0]  # (T, T)
    hot_abs = bd.get_epitope_mask(
        target_chains=target_seq.split("|"), binder_length=BINDER_LEN,
        target_hotspot_ids=hotspots, batch_size=1, device="cuda",
    )[0].nonzero(as_tuple=True)[0]
    exp_dist_hs = exp_pair[hot_abs].min(dim=0).values.cpu()  # (T,) 到最近热点期望距离

    def proxy_burial(cutoff: float, sep: int):
        pos = torch.arange(total_target)
        sep_mask = (pos[:, None] - pos[None, :]).abs() >= sep
        contact = (probs * (bins < cutoff)).sum(-1)[0]
        return (contact * sep_mask.to(contact.device)).sum(-1).cpu()

    burial10 = proxy_burial(10.0, 2)
    burial8 = proxy_burial(8.0, 2)

    # ── 晶体结构金标准 ───────────────────────────────────────────
    offsets = {"A": 17, "B": 32}  # complex 链标签 → author 偏移
    pdb_chains = parse_pdb_residues()
    dssp_rsa = {}
    try:
        dssp_rsa = run_dssp_rsa()
        print(f"[mask4zqk] DSSP RSA: {len(dssp_rsa)} 残基", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[mask4zqk] DSSP 失败 ({exc}), 仅用接触数金标准", flush=True)

    # seq 空间残基记录 + 身份校验
    res_list = []  # dict: idx(0-based flat), chain, seq_i, author, aa, cb, rsa
    n_mismatch = n_missing = 0
    flat = 0
    for cl in chain_labels:
        for i, aa in enumerate(seqs[cl], start=1):
            auth = offsets[cl] + i
            rec = pdb_chains.get(cl, {}).get(auth)
            ok = rec is not None and rec["aa"] == aa
            if rec is not None and rec["aa"] != aa:
                n_mismatch += 1
            if rec is None:
                n_missing += 1
            res_list.append({
                "idx": flat, "chain": cl, "seq_i": i, "author": auth,
                "aa": aa, "resolved": ok,
                "cb": rec["cb"] or rec["ca"] if rec else None,
                "rsa": dssp_rsa.get((cl, auth)),
            })
            flat += 1
    assert flat == total_target
    print(f"[mask4zqk] 身份校验: 未建模 {n_missing}, 错配 {n_mismatch} "
          f"(均不计入金标准指标)", flush=True)

    resv = [r for r in res_list if r["resolved"]]
    hot_set = set(hot_abs.tolist())

    def true_cb_near(r: dict, radius: float) -> bool:
        return any(
            r["cb"] is not None
            and res_list[h]["cb"] is not None
            and math.dist(r["cb"], res_list[h]["cb"]) < radius
            for h in hot_abs.tolist()
        )

    def true_cn(cutoff: float, sep: int) -> list[float]:
        out = []
        for r in resv:
            c = 0
            for q in resv:
                if abs(q["idx"] - r["idx"]) >= sep and math.dist(r["cb"], q["cb"]) < cutoff:
                    c += 1
            out.append(float(c))
        return out

    true_cn10 = true_cn(10.0, 2)
    rsa_pairs = [(r["rsa"], burial10[r["idx"]].item(), cn)
                 for r, cn in zip(resv, true_cn10) if r["rsa"] is not None]
    rsa_arr = [p[0] for p in rsa_pairs]
    prox_b = [p[1] for p in rsa_pairs]
    cn_at_rsa = [p[2] for p in rsa_pairs]
    print(f"\n[mask4zqk] 埋藏代理 vs 金标准 (n={len(rsa_arr)}):", flush=True)
    print(f"  Spearman(proxy_burial10, true_CN10) = "
          f"{spearman(prox_b, cn_at_rsa):.3f}", flush=True)
    print(f"  Spearman(proxy_burial10, DSSP RSA)  = "
          f"{spearman(prox_b, rsa_arr):.3f}  (期望为负)", flush=True)

    # ── 阈值扫描: 代理 mask vs 晶体同公式真值 mask ────────────────
    def true_accessible(q: float) -> set[int]:
        vals = sorted(c for c in true_cn10)
        k = max(0, min(len(vals) - 1, int(math.ceil(q * len(vals))) - 1))
        thr = vals[k]
        return {r["idx"] for r, c in zip(resv, true_cn10) if c < thr}

    print(f"\n[mask4zqk] 阈值扫描 (Jaccard vs 晶体同公式真值; 括号=纯度RSA≥20%):",
          flush=True)
    sweep = []
    for R in R_SWEEP:
        row = []
        for q in Q_SWEEP:
            near_p = {i for i in range(total_target) if exp_dist_hs[i] < R}
            acc_p = set((burial10 < torch.quantile(burial10, q)).nonzero(
                as_tuple=True)[0].tolist())
            mask_p = near_p & acc_p | hot_set
            near_t = {r["idx"] for r in resv if true_cb_near(r, R)}
            acc_t = true_accessible(q)
            mask_t = near_t & acc_t | {r["idx"] for r in resv if r["idx"] in hot_set}
            inter = len(mask_p & mask_t)
            jac = inter / len(mask_p | mask_t) if mask_p | mask_t else float("nan")
            pur = rsa_of(mask_p, res_list)
            row.append(f"{jac:.2f}({pur:.2f})")
            sweep.append({"R": R, "q": q, "size_proxy": len(mask_p),
                          "size_true": len(mask_t), "jaccard": jac,
                          "purity_rsa20": pur})
        print(f"  R={R:4.0f}: " + "  ".join(
            f"q{q}={v}" for q, v in zip(Q_SWEEP, row)), flush=True)

    # ── 默认配置 mask 明细 ────────────────────────────────────────
    default_set = set(mask_default[0].nonzero(as_tuple=True)[0].tolist())
    print(f"\n[mask4zqk] 默认阈值 (R=15, q=0.5) mask = {len(default_set)} 残基:",
          flush=True)
    detail = []
    for r in sorted(res_list, key=lambda x: (x["chain"], x["seq_i"])):
        sel = r["idx"] in default_set
        rsa_s = f"{r['rsa']:.2f}" if r["rsa"] is not None else "  ? "
        mark = "HOT" if r["idx"] in hot_set else (" * " if sel else "   ")
        if sel or r["idx"] in hot_set:
            print(f"  {mark} {r['chain']}{r['seq_i']:>4} {r['aa']} "
                  f"(author {r['author']})  RSA={rsa_s}  "
                  f"d_hs={exp_dist_hs[r['idx']].item():5.1f}Å", flush=True)
        detail.append({
            **{k: r[k] for k in ("idx", "chain", "seq_i", "author", "aa",
                                 "resolved")},
            "rsa": r["rsa"], "exp_dist_hotspot": float(exp_dist_hs[r["idx"]]),
            "proxy_burial10": float(burial10[r["idx"]]),
            "proxy_burial8": float(burial8[r["idx"]]),
            "in_default_mask": sel, "is_hotspot": r["idx"] in hot_set,
        })

    pur_default = rsa_of(default_set, res_list)
    print(f"\n[mask4zqk] 默认 mask 表面纯度 (RSA≥20%): {pur_default:.2f} | "
          f"平均 RSA: {mean_rsa(default_set, res_list):.2f}", flush=True)

    (OUT_DIR / "result.json").write_text(json.dumps({
        "task": TASK, "hotspots": hotspots, "binder_len": BINDER_LEN,
        "seed": SEED, "n_mismatch": n_mismatch, "n_missing": n_missing,
        "spearman_proxy_vs_trueCN": spearman(prox_b, cn_at_rsa),
        "spearman_proxy_vs_rsa": spearman(prox_b, rsa_arr),
        "default_mask": {"size": len(default_set), "purity_rsa20": pur_default,
                         "mean_rsa": mean_rsa(default_set, res_list)},
        "sweep": sweep, "detail": detail,
    }, indent=2))
    print(f"[mask4zqk] 产物 → {OUT_DIR / 'result.json'}", flush=True)


def torch_zeros_polya(batch: int, length: int):
    d = torch.zeros(batch, length, bd.AA_DIMS)
    d[..., 0] = 1.0
    return d.cuda()


def rsa_of(idxs: set[int], res_list: list[dict], threshold: float = 0.2) -> float:
    """mask 纯度: RSA ≥ threshold (表面) 的残基占比。"""
    vals = [res_list[i]["rsa"] for i in idxs
            if i < len(res_list) and res_list[i]["rsa"] is not None]
    if not vals:
        return float("nan")
    return sum(v >= threshold for v in vals) / len(vals)


def mean_rsa(idxs: set[int], res_list: list[dict]) -> float:
    vals = [res_list[i]["rsa"] for i in idxs
            if i < len(res_list) and res_list[i]["rsa"] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


if __name__ == "__main__":
    main()
