#!/usr/bin/env python3
"""
诊断实验: 多链靶热点引导 Stage-1 失效的根因定位。

背景 (2026-09-15): 4ZQK 双链靶 + 热点 (单/双表位 × w0.5/0.2 × 新旧编号) 4 配置
全部 Stage-1 崩溃 (iPTM 0.44-0.47, T/P 乱码), 同靶无热点 0.9077、单链 3RRQ+热点
0.8575 正常。mask/损失端/几何已逐一排除。

方法: 忠实复刻 binder_design.run_step (binder_design.py:1184-1262), 30 步早期窗口
(温度按 150 步日程计算), 对结构损失做分项 autograd.grad (autograd 线性 ⇒ 分项和
= 总梯度), 记录:
  - 分项归一化梯度方向占比 ‖w_i·masked_g_i‖ / Σ (与生产 normalized_gradient_tensor 同式)
  - structure_grad(含 epitope) vs 无 epitope 重算梯度的余弦
  - binder argmax 序列的 P/T/S/G/D/E 退化占比
  - 各损失值 + plm_loss
configs:
  A) 4ZQK 无热点 (对照, 应收敛)
  B) 4ZQK + ["A:S100"] (第 4 轮失败配置)
  C) 3RRQ + ["A:N85","A:R83"] (单链热点对照, 应收敛)

用法: CUDA_VISIBLE_DEVICES=<UUID> python diag_multichain_hotspot.py
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_pipeline_dir = Path(__file__).resolve().parents[2]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from Bio import SeqIO  # noqa: E402

# ---- Patch A (与 run_e2e 相同, COMPILE 开启的必要条件) ----
from transformers.models.esmfold2 import modeling_esmfold2_common as _common  # noqa: E402

_orig_addmm_residual = _common.Transition._addmm_residual

def _addmm_residual_ac(self, x, hidden):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _orig_addmm_residual(self, x, hidden)

_common.Transition._addmm_residual = _addmm_residual_ac

import esmcp.binder_design as bd  # noqa: E402

# COMPILE 图的 donated buffer 与本探针的 retain_graph=True 分项 backward 冲突;
# 且编译图峰值 ~20.9GiB 留不下分项 backward 的余量 → 探针用 eager
# (本机探针实证 eager+grad 12.7GiB 级, 步速约 2x 慢, 30 步可接受)。
bd.COMPILE = False
torch._functorch.config.donated_buffer = False

DATA = _pipeline_dir / "test" / "data"
OUT = _pipeline_dir / "test" / "module_results" / "diag_multichain_hotspot"
OUT.mkdir(parents=True, exist_ok=True)

RUN_STEPS = 30          # 早期窗口步数
SCHEDULE_STEPS = 150    # 温度日程按生产的 150 步算, 只跑前 RUN_STEPS 步
DEGEN_AA = set("PTSGDE")  # 乱码特征残基

def fasta_chains(p: Path) -> dict[str, str]:
    out = {}
    for r in SeqIO.parse(p, "fasta"):
        m = re.search(r"\|Chain (\S+)\|", r.description)
        out[m.group(1)] = str(r.seq).replace("X", "")
    return out

_chains = fasta_chains(DATA / "rcsb_pdb_4ZQK.fasta")
TARGET_4ZQK = f"{_chains['A']}|{_chains['B']}"
_ch3 = fasta_chains(DATA / "rcsb_pdb_3RRQ.fasta")
TARGET_3RRQ = _ch3["A"]

def aa_composition(logits: torch.Tensor, binder_length: int) -> dict:
    """binder 段 argmax 氨基酸组成 (logits [B, L_binder, 20], index+2 → TOKENS)."""
    inv = {v: k for k, v in bd.PROTEIN_1TO3.items()}
    amax = logits[0].argmax(dim=-1)  # [L]
    letters = [inv[bd.TOKENS[int(i) + 2]] for i in amax]
    n = len(letters)
    degen = sum(1 for a in letters if a in DEGEN_AA)
    top1_frac = max(letters.count(a) for a in set(letters)) / n
    return {"degen_frac": degen / n, "top1_aa_frac": top1_frac,
            "head": "".join(letters[:20])}

def probe_config(tag: str, target_seq: str, hotspots: list[str] | None,
                 app, seed: int = 0, batch_size: int = 1,
                 epitope_weight: float | None = None) -> dict:
    if epitope_weight is not None:
        bd.LOSS_WEIGHTS["epitope"] = epitope_weight
    log = lambda s: print(f"[{tag}] {s}", flush=True)
    binder_seq = bd.BINDER_PROMPT_FACTORIES["minibinder"].sample(seed=seed)
    assert "#" in binder_seq
    binder_length = len(binder_seq)
    target_one_hot = bd.build_target_one_hot(target_seq, device="cuda")

    with bd.seed_context(seed), torch.device("cuda"):
        logits = bd.build_initial_soft_sequence_logits(binder_seq, batch_size)
        gmask = bd.build_gradient_mask(binder_seq, batch_size)

    optimizer = torch.optim.SGD([logits], lr=bd.LEARNING_RATE)
    records: list[dict] = []
    log(f"binder_len={binder_length} target_len={len(target_seq.replace('|',''))} "
        f"hotspots={hotspots}")

    for step in range(RUN_STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        random.seed(seed + step)
        inversion_model = list(app.inversion_models.values())[
            random.randint(0, len(app.inversion_models) - 1)]
        temperature = bd.TEMPERATURE_MIN + (1 - bd.TEMPERATURE_MIN) * 0.5 * (
            1 + math.cos(math.pi * (step + 1) / SCHEDULE_STEPS))
        design = F.softmax(logits / temperature, dim=-1)

        fold = bd.fold_and_get_distogram(
            inversion_model, target_seq, target_one_hot, design,
            num_loops=1, num_sampling_steps=1, calculate_confidence=False,
            seed=seed + step,
        )
        losses = bd.compute_structure_losses(
            fold["distogram_logits"], binder_length,
            target_sequence=target_seq, target_hotspot_ids=hotspots,
            epitope_contact_distance=12.0,
        )
        structure_loss = losses["total_loss"]

        # 分项梯度 (autograd 线性: Σ w_i·g_i == 结构总梯度; 末项后释放图)
        terms = ["intra_contact_loss", "inter_contact_loss", "glob_loss"]
        if hotspots:
            terms.append("epitope_loss")
        weights = {"intra_contact_loss": bd.LOSS_WEIGHTS["intra_contact"],
                   "inter_contact_loss": bd.LOSS_WEIGHTS["inter_contact"],
                   "glob_loss": bd.LOSS_WEIGHTS["glob"],
                   "epitope_loss": bd.LOSS_WEIGHTS["epitope"]}
        dg = fold["distogram_logits"]
        # 分项梯度在 distogram 空间计算 (同一线性 loss 图, 经同一模型 Jacobian
        # 映射到 logits 空间 → 相对占比可用作方向分解的代理); logits 空间只做
        # 生产同式的一次总 backward, 避免 retain 多次折叠图导致的 OOM。
        term_norms, dg_grads = {}, {}
        for name in terms:
            g = torch.autograd.grad(
                (weights[name] * losses[name].mean()), dg, retain_graph=True)[0]
            dg_grads[name] = g
            term_norms[name] = float(torch.linalg.norm(
                g.reshape(g.shape[0], -1), dim=-1).mean().item())
        denom = sum(term_norms.values()) + 1e-9
        dir_frac = {k: v / denom for k, v in term_norms.items()}

        # 有/无 epitope 的结构梯度方向余弦 (distogram 空间; 总梯度=分项加和)
        cos_epi = None
        if hotspots:
            g_noepi = torch.autograd.grad(
                (bd.LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
                 + bd.LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
                 + bd.LOSS_WEIGHTS["glob"] * losses["glob_loss"]).mean(),
                dg, retain_graph=True)[0]
            g_all = sum(dg_grads.values())
            cos_epi = float(F.cosine_similarity(
                g_all.flatten().unsqueeze(0), g_noepi.flatten().unsqueeze(0)).item())

        # plm 梯度 (生产同式)
        design2 = F.softmax(logits / temperature, dim=-1)
        score_mask = gmask.sum(dim=-1) > 0
        with bd.seed_context(seed + step):
            plm_loss = bd.compute_esmc_pseudoperplexity_nll(
                esmc_model=app.esmc_model, binder_design=design2,
                score_mask=score_mask, batch_size=bd.LM_LOSS_BATCH_SIZE,
                n_passes=bd.LM_MASK_PASSES)
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]
        # 生产同式: 结构总梯度 (logits 空间, 最后一次 backward, 释放折叠图)
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]
        logits.grad = bd.normalized_gradient_tensor(structure_grad, gmask) + (
            0.15 * bd.normalized_gradient_tensor(plm_grad, gmask))
        for gparam in optimizer.param_groups:
            gparam["lr"] = bd.LEARNING_RATE * temperature
        optimizer.step()

        rec = {
            "step": step, "T": round(temperature, 4),
            "intra": float(losses["intra_contact_loss"].mean()),
            "inter": float(losses["inter_contact_loss"].mean()),
            "glob": float(losses["glob_loss"].mean()),
            "epitope": (float(losses["epitope_loss"].mean())
                        if hotspots else None),
            "plm": float(plm_loss.mean()),
            "dir_frac": {k: round(v, 4) for k, v in dir_frac.items()},
            "cos_with_vs_without_epitope": cos_epi,
            **{f"argmax_{k}": v for k, v in
               aa_composition(logits.detach(), binder_length).items()},
            "step_s": round(time.time() - t0, 1),
        }
        records.append(rec)
        if step % 5 == 0 or step == RUN_STEPS - 1:
            log(f"step {step:3d} T={temperature:.3f} intra={rec['intra']:.2f} "
                f"inter={rec['inter']:.2f} glob={rec['glob']:+.2f} "
                f"epi={rec['epitope'] if rec['epitope'] is None else round(rec['epitope'],2)} "
                f"plm={rec['plm']:.2f} dir={rec['dir_frac']} "
                f"cos={cos_epi} degen={rec['argmax_degen_frac']:.2f} head={rec['argmax_head']}")

    summary = {
        "tag": tag, "hotspots": hotspots,
        "epitope_weight": epitope_weight
        if epitope_weight is not None else bd.LOSS_WEIGHTS["epitope"],
        "binder_len": binder_length,
        "records": records,
        "note": "分项方向占比: 归一化后各项 ‖w·g‖ 占比; cos: 含/不含 epitope 的结构梯度方向余弦",
    }
    (OUT / f"probe_{tag}.json").write_text(json.dumps(summary, indent=2))
    log(f"完成 → {OUT / f'probe_{tag}.json'}")
    return summary


def main() -> None:
    # 模型装配 = ESMFold2Design.load() 去 hero critics (Stage-1 循环不需要)
    bd.ESMFold2Design.hero_critic_hf_paths = []
    app = bd.ESMFold2Design()
    t0 = time.time()
    app.load(use_scaling_critics=False)
    print(f"[setup] 模型加载 {time.time()-t0:.0f}s "
          f"(inversion={list(app.inversion_models)})")

    all_summary = {}
    # 低权重假设验证 (用户方案): w 大幅低于 0.5 时 epitope 方向占比随权重
    # 线性回落, 折叠损失重新主导 → 退化盆地应不可达 (intra 恢复下降)
    all_summary["D_4ZQK_AS100_w005"] = probe_config(
        "D_4ZQK_AS100_w005", TARGET_4ZQK, ["A:S100"], app, epitope_weight=0.05)
    all_summary["E_4ZQK_AS100_w010"] = probe_config(
        "E_4ZQK_AS100_w010", TARGET_4ZQK, ["A:S100"], app, epitope_weight=0.1)
    (OUT / "summary.json").write_text(json.dumps(all_summary, indent=2))
    print(f"[diag] 全部完成 → {OUT}")


if __name__ == "__main__":
    main()
