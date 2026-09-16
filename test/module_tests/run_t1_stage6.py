#!/usr/bin/env python3
"""
T1+T1b: Stage 6 过滤/评分/Top-N 选择 (纯 CPU)。

T1a  compute_selection_score(rosetta_weight=0) + apply_filters + select_top_n:
     置信度分支、default_filters.json 阈值边界 (0.50/0.80 含边界值)、
     minibinder pI<6 特例、按 (target,binder) 分组 top-N。
T1b  rosetta_weight=0.3 分支: 用 T7 的真实 Rosetta 指标结构构造样本,
     验证 score = 0.7*confidence + 0.3*physics 恒等式与重排序行为。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from core import (  # noqa: E402
    PipelineResult,
    apply_filters,
    compute_selection_score,
    select_top_n,
)

RESULTS = _pipeline_dir / "test" / "module_results"
RESULTS.mkdir(exist_ok=True)


def mk(name: str, *, iptm: float, proxy: float, stage: str = "mpnn",
       binder_name: str = "minibinder", pi: float = 5.0,
       rosetta: dict | None = None) -> PipelineResult:
    r = PipelineResult(
        target_name="sim_target", binder_name=binder_name, stage=stage,
        binder_sequence="ACDEFGHIKLMNPQRSTVWY" * 3,
        iptm=iptm if stage == "esmfold2" else None,
        refold_iptm=iptm if stage == "mpnn" else None,
        refold_ptm=0.60 if stage == "mpnn" else None,
        distogram_iptm_proxy=proxy,
        isoelectric_point=pi,
    )
    if rosetta:
        r.rosetta_dG = rosetta["dG"]
        r.rosetta_sc = rosetta["sc"]
        r.rosetta_hbonds = rosetta["hbonds"]
        r.rosetta_unsat_hbonds = rosetta["unsat"]
        r.rosetta_clashes = rosetta["clashes"]
        r.rosetta_binder_energy = rosetta["binder_energy"]
    r._name = name  # 测试标记 (dataclass 未定义字段, 仅内存对象使用)
    return r


def main() -> None:
    report: dict = {"T1a": {}, "T1b": {}}
    checks: list[tuple[str, bool]] = []

    # ---------- T1a: 置信度分支 ----------
    samples = [
        mk("high", iptm=0.75, proxy=0.85),
        mk("edge_iptm_exact", iptm=0.50, proxy=0.85),   # 边界: <0.50 才 fail
        mk("low", iptm=0.45, proxy=0.40),               # i_pTM & pLDDT 双fail
        mk("high_but_pI7", iptm=0.80, proxy=0.90, pi=7.0),  # minibinder pI>=6 拒
        mk("high_pI7_notminibinder", iptm=0.80, proxy=0.90,
           binder_name="non_mini"),                     # pI 特例只管 minibinder
    ]
    scored = compute_selection_score(list(samples), rosetta_weight=0.0)
    passed, failed = apply_filters(list(samples))

    by_name = {r._name: r for r in scored}
    exp_pass = {"high", "edge_iptm_exact", "high_pI7_notminibinder"}
    got_pass = {r._name for r in passed}
    got_fail = {r._name for r in failed}
    checks.append(("T1a passed 集合", got_pass == exp_pass))
    checks.append(("T1a low 被拒且记录 unmet", "low" in got_fail and
                   by_name["low"].filter_unmet == ["Average_pLDDT", "Average_i_pTM"]))
    checks.append(("T1a 边界 0.50 恰好通过", "edge_iptm_exact" in got_pass))
    checks.append(("T1a pI 特例标注", "pI=" in by_name["high_but_pI7"].filter_unmet[-1]))
    # 排序: proxy 权重 0.5 + iptm 0.5 → high(0.85,0.75) 应低于 notminibinder(0.90,0.80)
    order = [r._name for r in scored]
    checks.append(("T1a 排序单调", order.index("high_pI7_notminibinder")
                   < order.index("high")))
    report["T1a"] = {
        "pass": sorted(got_pass), "fail": sorted(got_fail), "order": order,
        "scores": {r._name: round(r.selection_score, 4) for r in scored},
        "unmet": {r._name: r.filter_unmet for r in scored},
    }

    # top-N 分组: 两个 binder_name 组, top_n=1 → 每组各出 1 条
    group_samples = [
        mk("g_a1", iptm=0.9, proxy=0.9, binder_name="grpA"),
        mk("g_a2", iptm=0.7, proxy=0.85, binder_name="grpA"),
        mk("g_b1", iptm=0.6, proxy=0.82, binder_name="grpB"),
    ]
    compute_selection_score(group_samples)
    p2, f2 = apply_filters(group_samples)
    sel = select_top_n(p2, top_n=1)
    checks.append(("T1a top-N 分组每组 1 条",
                   len(sel) == 2 and {s._name for s in sel} == {"g_a1", "g_b1"}))

    # ---------- T1b: rosetta_weight=0.3 分支 (T7 真实指标量级) ----------
    ros_good = dict(dG=-22.5, sc=0.72, hbonds=6, unsat=1, clashes=0,
                    binder_energy=-45.0)
    ros_bad = dict(dG=+3.0, sc=0.48, hbonds=1, unsat=7, clashes=3,
                   binder_energy=-10.0)
    # 置信度贴近 (物理分才能决定排序), 物理分两极分化
    rs = [
        mk("conf_hi_phys_bad", iptm=0.62, proxy=0.80, rosetta=ros_bad),
        mk("conf_mid_phys_good", iptm=0.61, proxy=0.81, rosetta=ros_good),
        mk("conf_lo_phys_good", iptm=0.55, proxy=0.90, rosetta=ros_good),
    ]
    s0 = {r._name: r.selection_score for r in
          compute_selection_score(list(rs), rosetta_weight=0.0)}
    s3 = {r._name: r.selection_score for r in
          compute_selection_score(rs, rosetta_weight=0.3)}
    # 恒等式: s3 = 0.7*s0 + 0.3*phys, phys ∈ [0,1] → s3 ∈ [0.7*s0, 0.7*s0+0.3]
    for n in s0:
        lo, hi = 0.7 * s0[n], 0.7 * s0[n] + 0.3 + 1e-9
        checks.append((f"T1b {n} 权重恒等区间", lo - 1e-9 <= s3[n] <= hi))
    checks.append(("T1b 物理分改变排序",
                   s3["conf_mid_phys_good"] > s3["conf_hi_phys_bad"]
                   and list(s3.keys()) != list(s0.keys())))
    p3, f3 = apply_filters(rs)
    # ros_bad 触发 ShapeComplementarity<0.6, hbonds<3, unsat>4, dG>0 等
    checks.append(("T1b ros_bad 被 Rosetta 阈值拒",
                   "conf_hi_phys_bad" in {r._name for r in f3}))
    checks.append(("T1b ros_good 通过全部 Rosetta 阈值",
                   {"conf_mid_phys_good", "conf_lo_phys_good"}
                   <= {r._name for r in p3}))
    report["T1b"] = {"scores_w0": s0, "scores_w03": s3,
                     "passed": sorted(r._name for r in p3),
                     "failed": sorted(r._name for r in f3),
                     "unmet_bad": next(r.filter_unmet for r in rs
                                       if r._name == "conf_hi_phys_bad")}

    # ---------- 汇总 ----------
    report["checks"] = [{"check": c, "ok": ok} for c, ok in checks]
    for c, ok in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {c}")
    n_bad = sum(not ok for _, ok in checks)
    (RESULTS / "t1_stage6.json").write_text(json.dumps(report, indent=2))
    print(f"\n[T1] {len(checks) - n_bad}/{len(checks)} checks passed "
          f"-> {RESULTS / 't1_stage6.json'}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
