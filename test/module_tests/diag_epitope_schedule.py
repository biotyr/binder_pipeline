#!/usr/bin/env python3
"""
诊断实验: epitope 权重驼峰调度 — 高位拉近, 触发后衰减持位。

用户假设: 弱权重 (0.05) 下热点信号抓不住 binder (inter 主导梯度把 binder
拖离热点), 有效优化步数被浪费; 强权重 (0.5) 则退化解崩溃。
方案: 阶段1 w=0.5 拉近 → 触发后指数衰减 → 阶段2 w=0.1 持位+精修。

触发: epitope ≤ 1.0 且 intra ≤ 3.5, 连续 25 步 (护栏挡退化解假触发:
探针 B 中 w0.5 退化解 epi 0.84 但 intra 4.76↑); 截止兜底 step 60。
步数 150 (已证最优日程), seed=1 (最佳 founder seed), 双热点 B:S30/A:S100。

只跑 Stage 1 + Stage 2 质量门 (判定点); 产物写独立目录, 不动正式结果。
基线: 常权 0.05 → iPTM 0.8245 (过门); w0.5 常权 → 0.44-0.47 (崩)。
"""
from __future__ import annotations

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

import torch  # noqa: E402
from Bio import SeqIO  # noqa: E402

import run_e2e  # noqa: E402,F401  (Patch A + pyrosetta.init + hero 限 1)
import esmcp.binder_design as bd  # noqa: E402
from esmcp.binder_design import ESMFold2Design  # noqa: E402

# ---- 调度参数 ----
W_START = 0.5
W_FLOOR = 0.1
THETA = 1.0
INTRA_GUARD = 3.5
SUSTAIN_STEPS = 25
DEADLINE_STEP = 60
TAU = 10.0
STEPS = 150
SEED = 1
HOTSPOTS = ["B:S30", "A:S100"]

_orig_csl = bd.compute_structure_losses


class EpitopeSchedule:
    """按步调度的 epitope 权重状态机 (run_step 每步恰调用一次
    compute_structure_losses, 调用计数即步数)。"""

    def __init__(self):
        self.calls = 0
        self.state = "phase1_high"
        self.trigger_at = None
        self.sustain = 0
        self.log: list[dict] = []

    def current_w(self) -> float:
        if self.state == "phase1_high":
            return W_START
        t = self.calls - self.trigger_at
        return W_FLOOR + (W_START - W_FLOOR) * math.exp(-t / TAU)

    def __call__(self, *a, **kw):
        w = self.current_w()
        bd.LOSS_WEIGHTS["epitope"] = w
        losses = _orig_csl(*a, **kw)
        step = self.calls
        self.calls += 1
        epi = float(losses["epitope_loss"].mean())
        intra = float(losses["intra_contact_loss"].mean())
        inter = float(losses["inter_contact_loss"].mean())
        rec = {"step": step, "w": round(w, 4), "epi": epi, "intra": intra,
               "inter": inter, "state": self.state}
        if self.state == "phase1_high":
            if epi <= THETA and intra <= INTRA_GUARD:
                self.sustain += 1
            else:
                self.sustain = 0
            if self.sustain >= SUSTAIN_STEPS:
                self.state, self.trigger_at = "phase2_decay", step
                print(f"[sched] 触发 @step {step} (epi={epi:.2f} intra={intra:.2f}) "
                      f"→ 指数衰减至 {W_FLOOR}", flush=True)
            elif step >= DEADLINE_STEP:
                self.state, self.trigger_at = "phase2_decay", step
                print(f"[sched] 截止兜底 @step {step} (epi={epi:.2f} "
                      f"intra={intra:.2f}) → 衰减至 {W_FLOOR}", flush=True)
        elif step % 10 == 0:
            print(f"[sched] step {step:3d} w={w:.3f} epi={epi:.2f} "
                  f"intra={intra:.2f} inter={inter:.2f}", flush=True)
        self.log.append(rec)
        return losses


def main() -> None:
    sched = EpitopeSchedule()
    bd.compute_structure_losses = sched

    DATA = _pipeline_dir / "test" / "data"
    seqs = {}
    for r in SeqIO.parse(DATA / "rcsb_pdb_4ZQK.fasta", "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        seqs[m.group(1)] = str(r.seq).replace("X", "")
    target_seq = f"{seqs['A']}|{seqs['B']}"

    app = ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[sched] 模型加载 {time.time()-t0:.0f}s; "
          f"w: {W_START}→{W_FLOOR}, θ={THETA}, guard={INTRA_GUARD}, "
          f"deadline={DEADLINE_STEP}, steps={STEPS}, seed={SEED}", flush=True)

    t0 = time.time()
    best_sequences, trajectory, critic_results = app.design(
        target_name="4ZQK_multi", binder_name="minibinder",
        target_sequence=target_seq, is_antibody=False,
        seed=SEED, batch_size=1,
        target_hotspot_ids=HOTSPOTS, epitope_contact_distance=12.0,
    )
    print(f"[sched] Stage-1 完成 {time.time()-t0:.0f}s "
          f"(触发 @step {sched.trigger_at})", flush=True)

    # ---- Stage 2 质量门 ----
    from core import IntegratedPipeline, PipelineConfig
    cfg = PipelineConfig(verbose=True)
    pipeline = IntegratedPipeline(cfg)
    quality_passed, quality_failed = pipeline._quality_filter_critic_results(
        critic_results)
    gate = "PASS" if quality_passed else "FAIL"
    entry = (quality_passed or quality_failed)[0]
    print(f"\n[sched] 质量门: {gate} | iPTM={entry.get('iptm')} "
          f"(基线 0.8245; w0.5 常权 0.44-0.47)", flush=True)
    if quality_failed:
        print(f"[sched] unmet: {quality_failed[0].get('_unmet', '见上方 [Quality] 行')}",
              flush=True)

    out = _pipeline_dir / "test" / "module_results" / "diag_epitope_schedule"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "config": {"W_START": W_START, "W_FLOOR": W_FLOOR, "THETA": THETA,
                   "INTRA_GUARD": INTRA_GUARD, "SUSTAIN_STEPS": SUSTAIN_STEPS,
                   "DEADLINE_STEP": DEADLINE_STEP, "TAU": TAU, "STEPS": STEPS,
                   "SEED": SEED, "HOTSPOTS": HOTSPOTS},
        "trigger_at": sched.trigger_at,
        "gate": gate,
        "iptm": entry.get("iptm"),
        "ptm": entry.get("ptm"),
        "proxy": entry.get("distogram_iptm_proxy"),
        "designed_sequence": entry.get("designed_sequence", ""),
        "schedule_log": sched.log,
    }, indent=2))
    print(f"[sched] 产物 → {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
