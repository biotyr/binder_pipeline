"""
MPNN 重设计序列的 ESMFold2 重折叠筛选流水线。

两阶段筛选:
1. Scaling 快筛 — 1 步扩散, 仅 distogram, 廉价剔除弱结合候选
2. Hero 精测  — 200 步扩散, 全原子坐标 + iPTM/pTM/pLDDT

使用方式::

    from refold.screen import RefoldScreen

    screen = RefoldScreen(
        target_seq="MTIK...",
        scaling_models={"scaling_v1": scaling_model},
        hero_models={"hero_v1": hero_model},
    )
    # 阶段 1: 快筛
    passed = screen.scaling_filter(mpnn_sequences, threshold=0.5)
    # 阶段 2: 精测
    results = screen.hero_predict(passed)
    # 或一步完成
    results = screen.run(mpnn_sequences)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .refolder import refold_distogram, refold_full


@dataclass
class RefoldScreenConfig:
    """重折叠筛选的阈值配置。

    Attributes
    ----------
    scaling_threshold : float
        distogram_iptm_proxy 最低阈值, 默认 0.50。
    scaling_steps : int
        快筛扩散步数, 默认 1。
    hero_steps : int
        精确预测扩散步数, 默认 200。
    hero_loops : int
        精确预测回收迭代数, 默认 3。
    is_antibody : bool
        Binder 是否为抗体 (影响 CDR distogram 计算)。
    verbose : bool
        是否打印进度。
    """

    scaling_threshold: float = 0.50
    scaling_steps: int = 1
    hero_steps: int = 200
    hero_loops: int = 3
    is_antibody: bool = False
    verbose: bool = True


class RefoldScreen:
    """MPNN 序列 → ESMFold2 重折叠 → 质量评分的流水线。

    Parameters
    ----------
    target_seq : str
        Target 氨基酸序列。
    scaling_models : dict[str, Any]
        Scaling Critic 模型字典 ``{name: ESMFold2ExperimentalModel}``。
    hero_models : dict[str, Any]
        Hero Critic 模型字典 ``{name: ESMFold2ExperimentalModel}``。
    config : RefoldScreenConfig | None
    seed : int | None
        全局随机种子 (每个序列的 seed = base_seed + idx)。
    """

    def __init__(
        self,
        target_seq: str,
        scaling_models: dict[str, Any] | None = None,
        hero_models: dict[str, Any] | None = None,
        config: RefoldScreenConfig | None = None,
        seed: int | None = None,
    ):
        self.target_seq = target_seq
        self.scaling_models = scaling_models or {}
        self.hero_models = hero_models or {}
        self.config = config or RefoldScreenConfig()
        self.seed = seed

    # ---- 阶段 1: Scaling 快筛 ----

    def scaling_filter(
        self,
        sequences: list[str],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """用 Scaling Critic (1 步扩散) 快速筛选候选序列。

        每条序列被所有 scaling_models 分别评分,
        取 distogram_iptm_proxy 最高值作为该序列的得分。

        Parameters
        ----------
        sequences : list[str]
            MPNN 生成的 binder 序列列表。
        threshold : float | None
            distogram_iptm_proxy 最低阈值。None 使用 config 默认值。

        Returns
        -------
        list[dict]
            通过筛选的序列, 每项包含:
            - "sequence": str
            - "distogram_iptm_proxy": float (最佳)
            - "best_scaling_model": str
            - "all_scores": dict[str, float]  (每个 model 的得分)
        """
        threshold = threshold if threshold is not None else self.config.scaling_threshold

        if not self.scaling_models:
            # 无 scaling models → 全部通过
            if self.config.verbose:
                print("[Scaling] No scaling models — passing all sequences.")
            return [
                {
                    "sequence": seq,
                    "distogram_iptm_proxy": float("nan"),
                    "best_scaling_model": "none",
                    "all_scores": {},
                }
                for seq in sequences
            ]

        passed: list[dict[str, Any]] = []
        n_total = len(sequences)

        for idx, binder_seq in enumerate(sequences):
            all_scores: dict[str, float] = {}
            best_score = -1.0
            best_model = ""

            for model_name, model in self.scaling_models.items():
                try:
                    fold = refold_distogram(
                        model,
                        self.target_seq,
                        binder_seq,
                        num_sampling_steps=self.config.scaling_steps,
                        calculate_confidence=False,
                        seed=_step_seed(self.seed, idx),
                    )
                    from esmcp.binder_design import (
                        compute_distogram_iptm_proxy,
                    )
                    proxy = compute_distogram_iptm_proxy(
                        fold["distogram_logits"],
                        fold["target_length"],
                        binder_seq,
                        self.config.is_antibody,
                    )
                    score = proxy.get("distogram_iptm_proxy", -1.0)
                    all_scores[model_name] = score
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                    del fold
                except Exception as exc:
                    if self.config.verbose:
                        print(
                            f"[Scaling] Error on {model_name}/{binder_seq[:15]}...: "
                            f"{exc}"
                        )
                    all_scores[model_name] = float("nan")

            if best_score >= threshold:
                passed.append({
                    "sequence": binder_seq,
                    "distogram_iptm_proxy": best_score,
                    "best_scaling_model": best_model,
                    "all_scores": all_scores,
                })

            if self.config.verbose and (idx + 1) % 10 == 0:
                print(
                    f"[Scaling] {idx + 1}/{n_total}  "
                    f"passed={len(passed)}  "
                    f"last={best_score:.4f}"
                )

        if self.config.verbose:
            print(
                f"[Scaling] Done: {len(passed)}/{n_total} passed "
                f"(threshold={threshold:.2f})"
            )

        # 按得分降序
        passed.sort(key=lambda r: r["distogram_iptm_proxy"], reverse=True)
        return passed

    # ---- 阶段 2: Hero 精测 (多模型, 取最优 iPTM) ----

    def hero_predict(
        self,
        scaling_passed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """用 Hero Critics (200 步扩散) 精确预测结构 + 全量置信度。

        每条序列被所有 ``hero_models`` 分别预测, 取 iPTM 最高的作为最终结果。
        模型逐个运行 (单 model 跑完全量序列后释放, 再加载下一个),
        以控制显存峰值。

        .. note::
            调用后 ``self.hero_models`` 会被清空 (所有权释放)。
            如需再次调用, 请传入新的 dict。

        Parameters
        ----------
        scaling_passed : list[dict]
            ``scaling_filter()`` 的返回值 (或任意含 "sequence" key 的列表)。

        Returns
        -------
        list[dict]
            每项包含:
            - "sequence": str
            - "target_seq": str
            - "complex": ProteinComplex | None
            - "annotated_pdb": str | None   (含 REMARK 标注)
            - "iptm": float | None
            - "ptm": float | None
            - "plddt": float | None
            - "distogram_iptm_proxy": float
            - "cdr_distogram_iptm_proxy": float
            - "hero_model": str   (产生最优结果的那个模型名)
            - "scaling_score": float | None
            - "elapsed_s": float
            - "error": str | None   (仅在所有模型均失败时出现)
        """
        if not self.hero_models:
            raise ValueError("No hero models provided for full prediction.")

        best_per_idx: dict[int, dict[str, Any]] = {}
        n_total = len(scaling_passed)

        # 迭代 list(keys()) 以允许 pop 释放模型
        for model_name in list(self.hero_models.keys()):
            model = self.hero_models[model_name]
            if self.config.verbose:
                print(f"[Hero] Scoring {n_total} sequences with model: {model_name}")

            for idx, entry in enumerate(scaling_passed):
                binder_seq = entry["sequence"]
                t_start = time.time()

                try:
                    full = refold_full(
                        model,
                        self.target_seq,
                        binder_seq,
                        num_sampling_steps=self.config.hero_steps,
                        num_loops=self.config.hero_loops,
                        seed=_step_seed(self.seed, idx),
                        is_antibody=self.config.is_antibody,
                    )
                    iptm_val = full.get("iptm") or -1.0
                    prev_iptm = best_per_idx.get(idx, {}).get("iptm") or -1.0

                    if iptm_val > prev_iptm:
                        best_per_idx[idx] = {
                            "sequence": binder_seq,
                            "target_seq": self.target_seq,
                            "complex": full["complex"],
                            "annotated_pdb": full["annotated_pdb"],
                            "iptm": full["iptm"],
                            "ptm": full["ptm"],
                            "plddt": full["plddt"],
                            "distogram_iptm_proxy": full["distogram_iptm_proxy"],
                            "cdr_distogram_iptm_proxy": full["cdr_distogram_iptm_proxy"],
                            "hero_model": model_name,
                            "scaling_score": entry.get("distogram_iptm_proxy"),
                            "elapsed_s": round(time.time() - t_start, 1),
                        }
                        if self.config.verbose:
                            print(
                                f"[Hero] {idx + 1}/{n_total}  "
                                f"{binder_seq[:15]}...  iPTM={iptm_val:.4f}  "
                                f"model={model_name}"
                            )
                    del full

                except Exception as exc:
                    if self.config.verbose:
                        print(
                            f"[Hero] Error on {model_name}/{binder_seq[:15]}...: {exc}"
                        )
                    # 仅当该序列尚无任何成功结果时记录 error entry
                    if idx not in best_per_idx:
                        best_per_idx[idx] = {
                            "sequence": binder_seq,
                            "target_seq": self.target_seq,
                            "complex": None,
                            "annotated_pdb": None,
                            "iptm": None,
                            "ptm": None,
                            "plddt": None,
                            "distogram_iptm_proxy": float("nan"),
                            "cdr_distogram_iptm_proxy": float("nan"),
                            "hero_model": model_name,
                            "scaling_score": entry.get("distogram_iptm_proxy"),
                            "elapsed_s": time.time() - t_start,
                            "error": str(exc),
                        }

            # 释放该模型, 为下一个模型腾出显存 (RefoldScreen 是唯一持有者)
            self.hero_models.pop(model_name, None)
            del model
            torch.cuda.empty_cache()

        # 按索引还原顺序, 再按 iPTM 降序
        results = [best_per_idx[i] for i in sorted(best_per_idx)]
        results.sort(
            key=lambda r: r["iptm"] if r["iptm"] is not None else -1.0,
            reverse=True,
        )

        if self.config.verbose:
            n_ok = sum(1 for r in results if r["complex"] is not None)
            n_models = len(results)
            print(f"[Hero] Done: {n_ok}/{n_total} successful predictions "
                  f"({n_models} sequences)")

        return results

    # ---- 完整流水线 ----

    def run(
        self,
        sequences: list[str],
        scaling_threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """执行完整的两阶段重折叠筛选。

        Parameters
        ----------
        sequences : list[str]
            MPNN 生成的 binder 序列。
        scaling_threshold : float | None

        Returns
        -------
        hero_results : list[dict]
            通过 hero 预测的结果 (按 iPTM 降序)。
        scaling_failed : list[dict]
            未通过 scaling 快筛的序列。
        """
        # 阶段 1
        passed = self.scaling_filter(sequences, threshold=scaling_threshold)
        scaling_passed_seqs = {r["sequence"] for r in passed}
        scaling_failed = [
            {"sequence": s, "distogram_iptm_proxy": float("nan")}
            for s in sequences
            if s not in scaling_passed_seqs
        ]

        if not passed:
            if self.config.verbose:
                print("[RefoldScreen] No sequences passed scaling filter.")
            return [], scaling_failed

        # 阶段 2
        hero_results = self.hero_predict(passed)
        return hero_results, scaling_failed


def _step_seed(base_seed: int | None, step: int) -> int | None:
    """确定性种子衍生。"""
    if base_seed is None:
        return None
    return base_seed + step * 10007
