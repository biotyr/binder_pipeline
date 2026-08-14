"""
逆折叠序列优化模块 (ProteinMPNN-based inverse folding).

流程:
    ESMFold2 Hero Critic PDB (含 REMARK 质量指标)
      │
      ├── quality_filter.filter_pdb_by_quality()
      │   └── 读取 iPTM/pTM/pLDDT + CA clash → 预过滤
      │
      ├── mpnn_redesign.mpnn_redesign_binder()
      │   ├── hotspot_residues() 检测界面残基
      │   ├── fix_pos = target_chain + binder 界面残基
      │   └── ProteinMPNN 采样 N 条序列
      │
      └── (下游) AF2 折叠验证 + Rosetta 物理过滤
"""

from .quality_filter import (
    QualityFilterConfig,
    batch_filter_pdbs,
    filter_pdb_by_quality,
)
from .mpnn_redesign import (
    mpnn_redesign_binder,
    mpnn_redesign_from_critic_results,
)

__all__ = [
    # quality_filter
    "QualityFilterConfig",
    "filter_pdb_by_quality",
    "batch_filter_pdbs",
    # mpnn_redesign
    "mpnn_redesign_binder",
    "mpnn_redesign_from_critic_results",
]
