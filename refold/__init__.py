"""
ESMFold2 重折叠验证模块。

将逆折叠优化 (ProteinMPNN) 产生的新 binder 序列与 target
重新输入 ESMFold2 进行结构预测和质量评估。

两阶段策略:
    Scaling (1步扩散, distogram) → Hero (200步, 全原子 + iPTM)

典型用法::

    from refold import RefoldScreen, RefoldScreenConfig

    screen = RefoldScreen(
        target_seq="MTIK...",
        scaling_models=scaling_models,
        hero_models=hero_models,
    )
    hero_results, scaling_failed = screen.run(
        mpnn_sequences,
        scaling_threshold=0.5,
    )
    # hero_results 可直接送入 bindcraft_filter 进行物理评分
"""

from .refolder import (
    binder_seq_to_design,
    refold_distogram,
    refold_full,
)
from .screen import (
    RefoldScreen,
    RefoldScreenConfig,
)

__all__ = [
    # refolder
    "binder_seq_to_design",
    "refold_distogram",
    "refold_full",
    # screen
    "RefoldScreen",
    "RefoldScreenConfig",
]
