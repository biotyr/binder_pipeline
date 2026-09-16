#!/usr/bin/env python3
"""
T2: Stage 2 质量闸门边界测试。

用 core.py Stage 2 的同款配置 (iPTM 双阈值 0.50/0.70, pLDDT 0.70,
proxy 0.50, max_ca_clashes=0) 对模拟样本跑 filter_pdb_by_quality,
验证每个变体按预期理由通过/拒绝。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from inverse_optimize.quality_filter import (  # noqa: E402
    QualityFilterConfig,
    filter_pdb_by_quality,
)
from esmcp.binder_design import _asym_id_to_chain_label  # noqa: E402
from core import _derive_chain_metadata  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = _pipeline_dir / "test" / "module_results"
RESULTS.mkdir(exist_ok=True)

# ---- core.py Stage 2 的阈值复现 ----
CFG_SINGLE = QualityFilterConfig(
    min_iptm=0.50, min_binder_plddt=0.70,
    min_distogram_binding_confidence=0.50, max_ca_clashes=0, binder_chain="B",
)
CFG_MULTI = QualityFilterConfig(
    min_iptm=0.70, min_binder_plddt=0.70,
    min_distogram_binding_confidence=0.50, max_ca_clashes=0, binder_chain="C",
)

# (文件, 配置, 期望 passed, 期望理由关键词或 None)
CASES = [
    ("single_pass.pdb",      CFG_SINGLE, True,  None),
    ("single_fail_iptm.pdb", CFG_SINGLE, False, "iPTM"),
    ("single_fail_plddt.pdb", CFG_SINGLE, False, "pLDDT"),
    ("single_fail_proxy.pdb", CFG_SINGLE, False, "dist_binding_confidence"),
    ("multi_pass.pdb",       CFG_MULTI,  True,  None),
    ("multi_fail_iptm.pdb",  CFG_MULTI,  False, "iPTM"),
    ("multi_clash.pdb",      CFG_MULTI,  False, "clashes"),
]


def main() -> None:
    manifest = json.loads((HERE / "sim_manifest.json").read_text())
    rows, n_fail = [], 0

    # 链名推导一致性: 模拟的 binder 链名 == core.py 按 designed_sequence 推导
    for fname, _, _, _ in CASES:
        m = manifest[fname]
        target_chains = m["designed_sequence"].split("|")[:-1]
        _, binder_label = _derive_chain_metadata(target_chains)
        assert binder_label == m["binder_chain"], (
            f"{fname}: core 推导 binder={binder_label} != 模拟 {m['binder_chain']}"
        )
    print(f"[chain-labels] core._derive_chain_metadata 与模拟样本全部一致 "
          f"(含多链 binder≠B: {[_asym_id_to_chain_label(2)]})")

    for fname, cfg, expect_pass, expect_kw in CASES:
        qc = filter_pdb_by_quality(str(HERE / "sim_pdbs" / fname), config=cfg)
        ok = qc["passed"] == expect_pass and (
            expect_kw is None
            or any(expect_kw in r for r in qc["reasons"])
        )
        n_fail += not ok
        rows.append({
            "file": fname, "passed": qc["passed"], "expected_pass": expect_pass,
            "iptm": qc["iptm"], "binder_plddt": round(qc["binder_plddt"], 4),
            "distogram_binding_confidence": qc["distogram_binding_confidence"],
            "ca_clashes": qc["ca_clashes"],
            "reasons": qc["reasons"], "case_ok": ok,
        })
        status = "OK " if ok else "FAIL"
        reasons = "; ".join(qc["reasons"]) or "-"
        print(f"[{status}] {fname:24s} passed={qc['passed']!s:5s} "
              f"iptm={qc['iptm']} plddt={qc['binder_plddt']:.3f} "
              f"proxy={qc['distogram_binding_confidence']} "
              f"clash={qc['ca_clashes']} | {reasons[:80]}")

    (RESULTS / "t2_quality_gate.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[T2] {len(CASES) - n_fail}/{len(CASES)} cases behaved as expected "
          f"-> {RESULTS / 't2_quality_gate.json'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
