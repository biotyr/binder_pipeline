#!/usr/bin/env python3
"""
junction 对退化挽救路径的 P2 权重测试 (seed=1, FORCE_P2 步25 无条件进入,
P2=0.1/0.1, P3 门不变)。对照 3d (P2=0.25/0.2, 自然触发): 其 P2 下 plm
持续 4.5-4.7 至步 90、intra 横盘至 105、终局 pLDDT 63.8/熵 2.0。假说:
junction 对锚定自满足, 低 P2 锚定让 plm+intra 拉回漂移、更早折叠。
背景: e 轮 (同 P2 权重) 因直通未测到 P2; 原生权重无调度直跑 = 0.646/
pLDDT 83.7/熵 3.34 (锚定折叠全成功, 仅 iPTM 0.646<0.70)。

两阶段实验结论 (diag_inter_schedule_ii): 步25 intra 横盘触发撤权重后折叠恢复
(intra 4.6→2.75, 熵 2.34→3.02) 但门仍 FAIL (iPTM 0.481) — 折叠的 binder 没有
折回热点贴片 (epi last5 2.60, 热点 12.1/18.2Å 无接触), 残余权重不足以精修界面。

三阶段状态机 (基序发现 → 折叠巩固 → 界面精修):
  Phase 1 基序发现  (步 0-25):   epi/inter = 0.5/0.5
    步25 (次级 步35) 检查点: intra_last5 > 3.8 → Phase 2 (臂 i 路径 intra≈3.0 不触发)
  Phase 2 折叠巩固  (触发后):    epi=0.15, inter=0.1
    步 ≥50 起每 5 步评估: intra_last10_mean < 3.0 → Phase 3 (折叠完成信号)
  Phase 3 界面精修  (触发后):    epi/inter 恢复 0.5/0.5
安全性: Phase3 时 binder 已紧实 (glob≈0.08), 贴片信号做刚体对接而非柔性搭附,
组成漂移根源不存在 (臂 i 实证折叠态 + w0.5 = 协同 corr +0.92); Phase3 落在
退火尾部, 对接位形由 T→0 冻结。若 Phase3 永不触发则退化为两阶段行为 (兜底)。

用法: CUDA_VISIBLE_DEVICES=<GPU-UUID> python diag_inter_schedule3f_junction.py
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

HOTSPOTS = ["B:S30", "A:S100"]  # 原始交界对 (双双界面边缘): 热点选择归因实验
OUT_DIR = _pipeline_dir / "test" / "module_results" / "diag_inter_schedule3f_junction"

# ---- 状态机参数 ----
INTRA_TRIGGER_P2 = 3.8        # Phase1→2: last5_mean 高于此 = 折叠横盘
CHECKPOINT_STEPS_P2 = (25, 35)
PHASE2_WEIGHTS = {"inter_contact": 0.1, "epitope": 0.1}  # 参数迭代f: 低 P2 锚定
FORCE_P2 = True  # 步 25 无条件进 P2: 保证退化挽救路径被测到 (e 轮因直通未测到 P2)
PHASE3_MIN_STEP = 50          # Phase3 最早评估步
PHASE3_WINDOW = 10            # Phase2→3: last10_mean
PHASE3_INTRA = 3.0            #   低于此 = 折叠完成
PHASE3_WEIGHTS = {"inter_contact": 0.5, "epitope": 0.5}

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
             'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# ---- 状态机: 计数 wrapper 包 compute_structure_losses ----
_ORIG_CSL = bd.compute_structure_losses
STATE = {"call": 0, "phase": 1, "intra": [], "events": []}


def _set_weights(phase: int, step: int, why: str, weights: dict) -> None:
    STATE["phase"] = phase
    bd.LOSS_WEIGHTS.update(weights)
    STATE["events"].append({"step": step, "event": f"phase{phase} ({why})",
                            "weights": dict(bd.LOSS_WEIGHTS)})
    print(f"[sched3] 步 {step}: → Phase {phase} ({why}) weights={weights}",
          flush=True)


def _csl_sched3(distogram_logits, binder_length, **kw):
    STATE["call"] += 1
    result = _ORIG_CSL(distogram_logits, binder_length, **kw)
    STATE["intra"].append(float(result["intra_contact_loss"].mean()))
    step = STATE["call"] - 1
    if STATE["phase"] == 1 and step in CHECKPOINT_STEPS_P2:
        mean5 = sum(STATE["intra"][-5:]) / 5
        STATE["events"].append({"step": step, "event": "p1_checkpoint",
                                "intra_last5_mean": round(mean5, 3)})
        print(f"[sched3] 步 {step} P1检查点: intra_last5={mean5:.3f} "
              f"(阈值 {INTRA_TRIGGER_P2})", flush=True)
        if mean5 > INTRA_TRIGGER_P2 or FORCE_P2:
            _set_weights(2, step,
                         f"intra plateau {mean5:.2f}" +
                         (" [FORCED]" if FORCE_P2 else ""),
                         PHASE2_WEIGHTS)
        elif step == CHECKPOINT_STEPS_P2[-1]:
            STATE["events"].append({"step": step, "event": "stay_phase1"})
            print("[sched3] 次级检查未触发, 永驻 Phase 1 (折叠进行中)", flush=True)
    elif STATE["phase"] == 2 and step >= PHASE3_MIN_STEP and step % 5 == 0:
        mean10 = sum(STATE["intra"][-PHASE3_WINDOW:]) / PHASE3_WINDOW
        STATE["events"].append({"step": step, "event": "p2_checkpoint",
                                "intra_last10_mean": round(mean10, 3)})
        print(f"[sched3] 步 {step} P2检查点: intra_last10={mean10:.3f} "
              f"(阈值 {PHASE3_INTRA})", flush=True)
        if mean10 < PHASE3_INTRA:
            _set_weights(3, step, f"folded intra {mean10:.2f}", PHASE3_WEIGHTS)
    return result


bd.compute_structure_losses = _csl_sched3


# ---- 分析工具 ----

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
    """热点→binder 最近 Cβ/CA 距离。输出 PDB 为复合体顺序编号 (链内序号 +
    前序链长 = flat 序号); 残基字母比较需 3→1 字母转换。"""
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
        ok = rec is not None and THREE2ONE.get(rec["aa"]) == letter
        if not ok:
            out[hs] = {"found": False,
                       "observed": rec["aa"] if rec else None, "flat": flat}
            continue
        ref = rec["cb"] or rec["ca"]
        d = sorted(math.dist(ref, b["cb"] or b["ca"]) for b in binder
                   if b["cb"] or b["ca"])
        out[hs] = {"found": True, "min_cb": round(d[0], 1),
                   "n12": sum(1 for x in d if x < 12),
                   "n8": sum(1 for x in d if x < 8)}
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
    print(f"[sched3] 模型加载 {time.time()-t0:.0f}s; 臂 ii 三阶段调度, "
          f"hotspots={HOTSPOTS}", flush=True)

    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name="minibinder",
        target_sequence=target_seq, is_antibody=False,
        seed=1, batch_size=1,
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
           "scheduler": {"final_phase": STATE["phase"],
                         "events": STATE["events"],
                         "n_calls": STATE["call"]},
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT_DIR / "result_raw.json").write_text(json.dumps(res, indent=2))
    if pdb_text:
        (OUT_DIR / "design.pdb").write_text(pdb_text)
    print(f"\n[sched3] 门={res['gate']} iPTM={res['iptm']} pTM={res['ptm']} "
          f"proxy={res['proxy']}", flush=True)
    print(f"[sched3] 调度事件: {res['scheduler']['events']}", flush=True)

    try:
        res["seq_health"] = seq_health(binder_seq)
        res["plddt_profile"] = plddt_profile(pdb_text) if pdb_text else {}
        res["hotspot_contacts"] = hotspot_contacts(pdb_text, HOTSPOTS, target_seq)
        res["traj"] = traj_summary(trajectory)
        res["baselines"] = {
            "no_schedule_ii": {"iptm": 0.4434, "binder_plddt": 33.4,
                               "entropy": 2.34, "gate": "FAIL"},
            "two_phase_ii": {"iptm": 0.4809, "binder_plddt": 48.8,
                             "entropy": 3.02, "gate": "FAIL"},
            "full_coverage_ii": {"iptm": 0.7130, "gate": "PASS"}}
        print(f"[sched3] 健康={res['seq_health']} pLDDT={res['plddt_profile']}\n"
              f"[sched3] 热点接触={res['hotspot_contacts']}\n"
              f"[sched3] 轨迹={res['traj']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        res["analysis_error"] = repr(exc)
        print(f"[sched3] 分析失败 (原始结果已保存): {exc!r}", flush=True)
    (OUT_DIR / "result.json").write_text(json.dumps(res, indent=2))
    print(f"[sched3] 产物 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
