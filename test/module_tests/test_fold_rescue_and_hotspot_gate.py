#!/usr/bin/env python3
"""fold_rescue 调度状态机 + 热点几何核验门 的单元测试 (CPU, 无模型)。

覆盖 (2026-10-14 源码化验收):
1. update_fold_rescue_schedule — 横盘轨迹: 步25 触发 P2 (0.1/0.1), folding
   达标后触发 P3 (0.5/0.5); 健康轨迹: 双检查点通过 → stay_phase1 永驻;
   无热点调用 design_binder+fold_rescue_schedule 的报错逻辑不在本测试
   (需 GPU), 以状态机纯函数测试为主。
2. IntegratedPipeline._hotspot_contact_check — 用真实产物做夹具:
   seed4 design.pdb (双热点真接触) → True; seed3 design.pdb (热点放弃
   28-30Å, 假阳性过门案例) → False。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import esmcp.binder_design as bd  # noqa: E402
from core import IntegratedPipeline  # noqa: E402

_AAS = "ACDEFGHIKLMNPQRSTVWY"
M4ZQK = _AAS * 5 + _AAS[:15]  # 115 (占位, 仅长度参与 flat 换算)
M4ZQK_B = _AAS * 5 + _AAS[:18]  # 118
TARGET_SEQ = f"{M4ZQK}|{M4ZQK_B}"
assert (len(M4ZQK), len(M4ZQK_B)) == (115, 118)


def test_schedule_plateau_path():
    state = {"call": 0, "phase": 1, "intra": [], "events": []}
    weights = dict(bd.LOSS_WEIGHTS)
    intra = [4.6] * 25 + [4.2] * 25 + [3.4] * 20 + [2.9] * 80  # 横盘→缓降→达标
    for step in range(len(intra)):
        bd.update_fold_rescue_schedule(state, weights, step, intra[step])
    events = [e["event"] for e in state["events"]]
    assert "phase2" in events and "phase3" in events, events
    p2 = next(e for e in state["events"] if e["event"] == "phase2")
    p3 = next(e for e in state["events"] if e["event"] == "phase3")
    assert p2["step"] == 25, p2
    # 记录的是完整权重字典, 校验救援子集
    assert all(p2["weights"][k] == v for k, v in bd.FOLD_RESCUE_P2_WEIGHTS.items()), p2
    assert p3["step"] > p2["step"] and p3["step"] >= 50, p3
    assert all(p3["weights"][k] == v for k, v in bd.FOLD_RESCUE_P3_WEIGHTS.items()), p3
    print(f"[1] 横盘路径 OK: P2@{p2['step']} → P3@{p3['step']}, 权重转移正确")


def test_schedule_healthy_path():
    state = {"call": 0, "phase": 1, "intra": [], "events": []}
    weights = dict(bd.LOSS_WEIGHTS)
    # 健康轨迹 (臂 i 型): 步 10 起下降, 步 25 last5 < 3.8
    intra = [4.6, 4.1, 3.7, 3.6, 3.6, 3.5, 3.3, 3.1, 2.9, 2.8] * 15
    for step in range(len(intra)):
        bd.update_fold_rescue_schedule(state, weights, step, intra[step])
    events = [e["event"] for e in state["events"]]
    assert "phase2" not in events, events
    assert state["phase"] == 1 and events[-1] == "stay_phase1"
    assert weights["epitope"] == bd.LOSS_WEIGHTS["epitope"]  # 未被改
    print("[2] 健康路径 OK: stay_phase1, 权重全程原生")


def test_hotspot_gate():
    root = Path(__file__).resolve().parents[1] / "module_results" / \
        "diag_inter_schedule3g_junction"
    hs = ["B:S30", "A:S100"]
    ok4, detail4 = IntegratedPipeline._hotspot_contact_check(
        (root / "seed4" / "design.pdb").read_text(), hs, TARGET_SEQ)
    ok3, detail3 = IntegratedPipeline._hotspot_contact_check(
        (root / "seed3" / "design.pdb").read_text(), hs, TARGET_SEQ)
    assert ok4 is True, detail4
    assert ok3 is False, f"seed3 假阳性应被拒: {detail3}"
    print(f"[3] 热点几何门 OK: seed4 (真锚定) → True; "
          f"seed3 (假阳性 28-30Å) → False ({detail3[:60]}...)")


def main() -> None:
    test_schedule_plateau_path()
    test_schedule_healthy_path()
    test_hotspot_gate()
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
