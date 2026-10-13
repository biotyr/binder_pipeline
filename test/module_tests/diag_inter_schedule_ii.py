#!/usr/bin/env python3
"""
臂 ii 重测: intra 横盘触发的两阶段权重调度 (纯测试脚本, 不动源码)。

背景: 邻域 inter 模式下臂 ii (A:Q30+A:D73 单链贴片) 失败 —
iPTM 0.4434 / binder pLDDT 33.4 / 熵 2.34 / 热点未接触 (16.4/21.1Å)。
机制: 单贴片吸引子 → 柔性组成漂移 (plm 3.06→5.66, 步 25-55) → 冻结不可折叠。

本实验状态机 (阶段语义: 基序发现 → 折叠巩固):
  Phase 1 (步 0-25):   inter_nb=0.5, epitope=0.5  — 同上次臂 ii, 轨迹逐位可复现
  检查点 步25: last5_mean(intra) > 3.8 ?  → 是: 切 Phase 2
  次级检查 步35: 同判据 → 是: 切 Phase 2; 否: 永驻 Phase 1 (臂 i 路径,
                 其步 21-25 intra ≈ 3.0 < 3.8, 分析上不会误触发)
  Phase 2 (此后):     inter_nb=0.1, epitope=0.15 — 撤贴片追击压力,
                      intra/glob 主导折叠; 留残余信号锚住已生成的基序
预期: 臂 ii 在步 25 触发 (intra ≈ 4.5 > 3.8), Phase 2 恢复折叠;
      若 Phase 2 后 intra 仍不降 → 组成在步 25 已不可救, 界定干预窗上限。

实现: monkey-patch bd.compute_structure_losses (计数即步数), 检查点处改
bd.LOSS_WEIGHTS (源码每次调用时读取)。plm 权重 0.15 是 run_step 内字面量
无法无源码修改, 本轮不动 (单变量归因; 若 Phase 2 无效, plm 联动是下一杠杆)。
用法: CUDA_VISIBLE_DEVICES=<GPU-UUID> python diag_inter_schedule_ii.py
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

HOTSPOTS = ["A:Q30", "A:D73"]
OUT_DIR = _pipeline_dir / "test" / "module_results" / "diag_inter_schedule_ii"
# 状态机参数
INTRA_TRIGGER = 3.8        # last5_mean(intra) 高于此值 = 折叠横盘
CHECKPOINT_STEPS = (25, 35)  # 主/次检查点 (0-indexed 步号)
PHASE2_WEIGHTS = {"inter_contact": 0.1, "epitope": 0.15}

# ---- 状态机: 计数 wrapper 包 compute_structure_losses ----
_ORIG_CSL = bd.compute_structure_losses
STATE = {"call": 0, "phase": 1, "intra": [], "events": []}


def _switch(step: int, why: str) -> None:
    STATE["phase"] = 2
    bd.LOSS_WEIGHTS.update(PHASE2_WEIGHTS)
    STATE["events"].append({"step": step, "event": f"phase2 ({why})",
                            "weights": dict(bd.LOSS_WEIGHTS)})
    print(f"[sched] 步 {step}: 切 Phase 2 ({why}) → "
          f"{PHASE2_WEIGHTS}", flush=True)


def _csl_scheduled(distogram_logits, binder_length, **kw):
    STATE["call"] += 1
    result = _ORIG_CSL(distogram_logits, binder_length, **kw)
    STATE["intra"].append(float(result["intra_contact_loss"].mean()))
    n, step = STATE["call"], STATE["call"] - 1
    if STATE["phase"] == 1 and step in CHECKPOINT_STEPS:
        last5 = STATE["intra"][-5:]
        mean5 = sum(last5) / len(last5)
        STATE["events"].append({"step": step, "event": "checkpoint",
                                "intra_last5_mean": round(mean5, 3)})
        print(f"[sched] 步 {step} 检查点: intra_last5={mean5:.3f} "
              f"(阈值 {INTRA_TRIGGER})", flush=True)
        if mean5 > INTRA_TRIGGER:
            _switch(step, f"intra plateau {mean5:.2f}>{INTRA_TRIGGER}")
        elif step == CHECKPOINT_STEPS[-1]:
            STATE["events"].append({"step": step, "event": "stay_phase1"})
            print("[sched] 次级检查未触发, 永驻 Phase 1 (折叠进行中)", flush=True)
    return result


bd.compute_structure_losses = _csl_scheduled


# ---- 分析工具 (与 diag_inter_neighborhood_stage1 相同口径) ----

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
            per.setdefault(ln[21], []).append(float(ln[60:66]))
    return {ch: {"n": len(v), "mean": round(sum(v) / len(v), 1),
                 "min": round(min(v), 1)} for ch, v in per.items()}


def pearson(a, b) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return round(cov / (va * vb + 1e-12), 3) if va > 0 and vb > 0 else float("nan")


def traj_summary(traj: dict) -> dict:
    series = {}
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
    return out


def hotspot_contacts(pdb_text: str, hotspots: list[str],
                     target_seq: str) -> dict:
    """热点→binder 最近 Cβ/CA 距离; 输出 PDB 为复合体顺序编号 (链内序号
    + 前序链长 = flat 序号)。"""
    chain_lens = [len(c) for c in target_seq.split("|")]
    atoms, order = {}, []
    for ln in pdb_text.split("\n"):
        if not ln.startswith("ATOM") or ln[12:16].strip() not in ("CA", "CB"):
            continue
        ch, num = ln[21], int(ln[22:26])
        atoms.setdefault((ch, num), {"aa": ln[17:20].strip(), "cb": None, "ca": None})
        if ch not in order:
            order.append(ch)
        atoms[(ch, num)]["cb" if ln[12:16].strip() == "CB" else "ca"] = (
            float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    if not order:
        return {}
    binder = [v for (c, _), v in atoms.items() if c == order[-1]]
    out = {}
    for hs in hotspots:
        ch, rid = hs.split(":")
        letter, idx = rid[0], int(rid[1:])
        ch_i = order.index(ch) if ch in order else -1
        flat = sum(chain_lens[:ch_i]) + idx if ch_i >= 0 else idx
        rec = atoms.get((ch, flat))
        ok = rec is not None and rec["aa"] == letter
        if not ok:
            out[hs] = {"found": False, "observed": rec["aa"] if rec else None}
            continue
        ref = rec["cb"] or rec["ca"]
        d = sorted(math.dist(ref, b["cb"] or b["ca"]) for b in binder
                   if b["cb"] or b["ca"])
        out[hs] = {"found": True, "min_cb": round(d[0], 1),
                   "n12": sum(1 for x in d if x < 12)}
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA = _pipeline_dir / "test" / "data"
    seqs = {}
    for r in SeqIO.parse(DATA / "rcsb_pdb_4ZQK.fasta", "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        seqs[m.group(1)] = str(r.seq).replace("X", "")
    target_seq = f"{seqs['A']}|{seqs['B']}"

    app = ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[sched] 模型加载 {time.time()-t0:.0f}s; 臂 ii 重测 "
          f"(intra 横盘触发两阶段调度), hotspots={HOTSPOTS}", flush=True)

    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name="minibinder",
        target_sequence=target_seq, is_antibody=False,
        seed=0, batch_size=1,
        target_hotspot_ids=HOTSPOTS, epitope_contact_distance=12.0,
        inter_loss_mode="hotspot_neighborhood",
    )
    from core import IntegratedPipeline, PipelineConfig
    pipeline = IntegratedPipeline(PipelineConfig(verbose=True))
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    entry = (quality_passed or quality_failed)[0]
    binder_seq = (entry.get("designed_sequence") or "").split("|")[-1]
    pdb_text = entry.get("annotated_pdb") or ""

    res = {"hotspots": HOTSPOTS,
           "gate": "PASS" if quality_passed else "FAIL",
           "iptm": entry.get("iptm"), "ptm": entry.get("ptm"),
           "proxy": entry.get("distogram_iptm_proxy"),
           "binder_sequence": binder_seq,
           "scheduler": {"phase": STATE["phase"], "events": STATE["events"],
                         "n_calls": STATE["call"]},
           "elapsed_s": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result_raw.json").write_text(json.dumps(res, indent=2))
    if pdb_text:
        (OUT_DIR / "design.pdb").write_text(pdb_text)
    print(f"\n[sched] 门={res['gate']} iPTM={res['iptm']} pTM={res['ptm']} "
          f"proxy={res['proxy']}", flush=True)
    print(f"[sched] 调度事件: {STATE['events']}", flush=True)

    try:
        res["seq_health"] = seq_health(binder_seq)
        res["plddt_profile"] = plddt_profile(pdb_text) if pdb_text else {}
        res["hotspot_contacts"] = hotspot_contacts(pdb_text, HOTSPOTS, target_seq)
        res["traj"] = traj_summary(trajectory)
        res["prev_ii_baseline"] = {
            "iptm": 0.4434, "binder_plddt_mean": 33.4, "entropy": 2.34,
            "hotspot_min_cb": [16.4, 21.1], "note": "上次无调度臂 ii"}
        print(f"[sched] 健康={res['seq_health']} pLDDT={res['plddt_profile']}\n"
              f"[sched] 热点接触={res['hotspot_contacts']}\n"
              f"[sched] 轨迹={res['traj']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        res["analysis_error"] = repr(exc)
        print(f"[sched] 分析失败 (原始结果已保存): {exc!r}", flush=True)
    (OUT_DIR / "result.json").write_text(json.dumps(res, indent=2))
    print(f"[sched] 产物 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
