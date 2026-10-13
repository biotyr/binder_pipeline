#!/usr/bin/env python3
"""
平行模式 Stage-1 真实测试: inter_loss_mode="hotspot_neighborhood"。

4ZQK, minibinder 工厂 seed0 → 158aa, w_epi=0.5, 150 步 — 与崩溃基线
(diag_4zqk_mixed i_BS30_AE28, 全覆盖 inter) 唯一差异是 inter 域限定。
基线: i_BS30_AE28 → FAIL iPTM 0.4583, 序列 NRRPPP/SSS/GGG (熵 2.68);
      ii_AQ30_AD73 → PASS iPTM 0.7130; founder(w0.05) → iPTM 0.8245。
双臂:
  i_junction_NB  : B:S30 + A:E28 (交界边缘对, 毒点配置) — 期望脱毒
  ii_noniface_NB : A:Q30 + A:D73 (双非界面, 良性配置) — 期望不劣化
判读: 质量门 / iPTM / 序列健康度 / 逐链 pLDDT / 热点真实接触 / 拔河轨迹
(corr(epitope, inter), 各 loss 首末段均值)。
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_pipeline_dir = Path(__file__).resolve().parents[2]
for p in (str(_pipeline_dir), str(_pipeline_dir / "test" / "e2e")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Bio import SeqIO  # noqa: E402

import run_e2e  # noqa: E402,F401  (Patch A + pyrosetta.init + hero 限 1)
import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import ESMFold2Design  # noqa: E402

bd.LOSS_WEIGHTS["epitope"] = 0.5

ARMS = [
    ("i_junction_NB", ["B:S30", "A:E28"]),
    ("ii_noniface_NB", ["A:Q30", "A:D73"]),
]
OUT_ROOT = _pipeline_dir / "test" / "module_results" / "diag_inter_neighborhood_stage1"


def seq_health(seq: str) -> dict:
    cnt = collections.Counter(seq)
    H = -sum((c / len(seq)) * math.log2(c / len(seq)) for c in cnt.values())
    maxrun, cur = 1, 1
    for a, b in zip(seq, seq[1:]):
        cur = cur + 1 if a == b else 1
        maxrun = max(maxrun, cur)
    top2 = sum(c for _, c in cnt.most_common(2)) / len(seq)
    return {"entropy": round(H, 2), "max_homopolymer": maxrun,
            "top2_aa_frac": round(top2, 2)}


def plddt_profile(pdb_text: str) -> dict:
    per = {}
    for ln in pdb_text.split("\n"):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ch = ln[21]
            per.setdefault(ch, []).append(float(ln[60:66]))
    return {ch: {"n": len(v), "mean": round(sum(v) / len(v), 1),
                 "min": round(min(v), 1)} for ch, v in per.items()}


def pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return round(cov / (va * vb + 1e-12), 3) if va > 0 and vb > 0 else float("nan")


def traj_summary(traj: dict) -> dict:
    series: dict[str, list[float]] = {}
    for k in ("intra_contact_loss", "inter_contact_loss", "epitope_loss",
              "glob_loss", "plm_loss"):
        vals = [float(traj[s][k].mean()) for s in sorted(traj) if k in traj[s]]
        if vals:
            series[k] = vals
    out = {k: {"first5": round(sum(v[:5]) / 5, 3),
               "last5": round(sum(v[-5:]) / 5, 3),
               "min": round(min(v), 3)} for k, v in series.items()}
    out["corr_epitope_vs_inter"] = pearson(series.get("epitope_loss", []),
                                           series.get("inter_contact_loss", []))
    out["corr_epitope_vs_intra"] = pearson(series.get("epitope_loss", []),
                                           series.get("intra_contact_loss", []))
    return out


def hotspot_contacts(pdb_text: str, hotspots: list[str],
                     target_seq: str) -> dict:
    """预测复合体中各热点到 binder (链序最后一链) 的最近 Cβ/CA 距离。

    热点 ID 格式 ``<chain>:<残基字母><链内序号>`` (如 B:S30)。输出 PDB
    残基按复合体顺序编号 (链A 1-115, 链B 116-233, ...), 故链内序号需加
    前序链长度换算为 flat 序号; 残基字母校验不符时在 ±2 内找回 (仅告警)。
    """
    chain_lens = [len(c) for c in target_seq.split("|")]
    atoms: dict[tuple[str, int], dict] = {}
    chains_in_order: list[str] = []
    for ln in pdb_text.split("\n"):
        if not ln.startswith("ATOM"):
            continue
        el = ln[12:16].strip()
        if el not in ("CA", "CB"):
            continue
        ch, num, aa3 = ln[21], int(ln[22:26]), ln[17:20].strip()
        if (ch, num) not in atoms:
            atoms[(ch, num)] = {"aa": aa3, "cb": None, "ca": None}
            if ch not in chains_in_order:
                chains_in_order.append(ch)
        atoms[(ch, num)]["cb" if el == "CB" else "ca"] = (
            float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    if not chains_in_order:
        return {}
    binder_ch = chains_in_order[-1]
    binder = [v for (ch, _), v in atoms.items() if ch == binder_ch]
    out = {}
    for hs in hotspots:
        ch, rid = hs.split(":")
        aa_letter, idx = rid[0], int(rid[1:])
        ch_i = chains_in_order.index(ch) if ch in chains_in_order else -1
        flat = sum(chain_lens[:ch_i]) + idx if ch_i >= 0 else idx
        rec = atoms.get((ch, flat))
        note = ""
        if rec is not None and rec["aa"] != aa_letter:
            note = f"(residue mismatch: pdb={rec['aa']})"
            for k in range(1, 3):  # ±2 内找回
                rec = atoms.get((ch, flat + k)) or atoms.get((ch, flat - k))
                if rec is not None and rec["aa"] == aa_letter:
                    note = f"(offset {k:+d})"
                    break
                rec = None
        if rec is None or not (rec["cb"] or rec["ca"]):
            out[hs] = {"found": False, "note": note, "flat": flat}
            continue
        ref = rec["cb"] or rec["ca"]
        d = sorted(math.dist(ref, b["cb"] or b["ca"]) for b in binder
                   if b["cb"] or b["ca"])
        out[hs] = {"found": True, "pdb_aa": rec["aa"], "note": note,
                   "min_cb_dist": round(d[0], 1),
                   "n_binder_within_12A": sum(1 for x in d if x < 12.0)}
    return out


def run_one(tag: str, hotspots: list[str], app, target_seq: str) -> dict:
    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name="minibinder",
        target_sequence=target_seq, is_antibody=False,
        seed=0, batch_size=1,
        target_hotspot_ids=hotspots, epitope_contact_distance=12.0,
        inter_loss_mode="hotspot_neighborhood",
    )
    from core import IntegratedPipeline, PipelineConfig
    pipeline = IntegratedPipeline(PipelineConfig(verbose=True))
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    entry = (quality_passed or quality_failed)[0]
    gate = "PASS" if quality_passed else "FAIL"
    binder_seq = (entry.get("designed_sequence") or "").split("|")[-1]

    # 防崩溃: 原始结果立即落盘, 分析项各自 try/except, 单项失败不丢运行。
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    res = {"tag": tag, "hotspots": hotspots, "gate": gate,
           "iptm": entry.get("iptm"), "ptm": entry.get("ptm"),
           "proxy": entry.get("distogram_iptm_proxy"),
           "binder_sequence": binder_seq,
           "elapsed_s": round(time.time() - t0, 1)}
    (out / "result_raw.json").write_text(json.dumps(res, indent=2))
    if entry.get("annotated_pdb"):
        (out / "design.pdb").write_text(entry["annotated_pdb"])
    pdb_text = entry.get("annotated_pdb") or ""
    print(f"\n[NB-stage1] {tag}: 门={gate} iPTM={res['iptm']} pTM={res['ptm']} "
          f"proxy={res['proxy']}", flush=True)

    try:
        res["seq_health"] = seq_health(binder_seq)
        res["plddt_profile"] = plddt_profile(pdb_text) if pdb_text else {}
        res["hotspot_contacts"] = hotspot_contacts(pdb_text, hotspots, target_seq)
        res["traj"] = traj_summary(trajectory)
        print(f"[NB-stage1] 健康={res['seq_health']} "
              f"pLDDT={res['plddt_profile']}\n"
              f"[NB-stage1] 热点接触={res['hotspot_contacts']}", flush=True)
        print(f"[NB-stage1] 拔河: "
              f"corr(epi,inter)={res['traj']['corr_epitope_vs_inter']} "
              f"corr(epi,intra)={res['traj']['corr_epitope_vs_intra']}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        res["analysis_error"] = repr(exc)
        print(f"[NB-stage1] 分析失败 (原始结果已保存): {exc!r}", flush=True)
    print(f"[NB-stage1] binder: {binder_seq[:60]}...", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    DATA = _pipeline_dir / "test" / "data"
    seqs = {}
    for r in SeqIO.parse(DATA / "rcsb_pdb_4ZQK.fasta", "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        seqs[m.group(1)] = str(r.seq).replace("X", "")
    target_seq = f"{seqs['A']}|{seqs['B']}"

    app = ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[NB-stage1] 模型加载 {time.time()-t0:.0f}s; "
          f"inter=hotspot_neighborhood, w_epi=0.5, factory(158), seed0", flush=True)

    results = {}
    for tag, hotspots in ARMS:
        results[tag] = run_one(tag, hotspots, app, target_seq)
    print("\n[NB-stage1] 汇总 (基线: i=FAIL 0.4583 熵2.68 | ii=PASS 0.7130):",
          flush=True)
    for tag, res in results.items():
        print(f"  {tag}: 门={res['gate']} iPTM={res['iptm']} "
              f"熵={res['seq_health']['entropy']} "
              f"corr(epi,inter)={res['traj']['corr_epitope_vs_inter']}", flush=True)


if __name__ == "__main__":
    main()
