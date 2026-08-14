# ESMFold2 × ProteinMPNN × BindCraft 集成 Pipeline

端到端 Binder 设计流程：ESMFold2 梯度设计 → 逆折叠优化 → 重折叠验证 → 物理化学过滤。

## 目录结构

```
pipeline/
├── README.md                              ← 本文件
│
├── core.py                                ← 核心模块 (纯 Python, 零 Modal 依赖)
│   ├── PipelineConfig / PipelineResult
│   ├── compute_selection_score / apply_filters / select_top_n
│   ├── IntegratedPipeline (6 阶段 run())
│   └── _collect_results / _save_outputs
│
├── run_modal.py                           ← Modal 远程部署适配器
│   └── _get_modal_image() / IntegratedPipelineModal / main()
│
├── run_local.py                           ← 纯本地运行入口
│   └── argparse → IntegratedPipeline(config).run(...)
│
├── esmcp/                                 ← ESMFold2 设计引擎
│   └── binder_design.py                   ← 核心: 150步 SGD + Inversion/Critic 评分
│                                             - pdb_with_quality_remarks() 将 iPTM/pTM
│                                               嵌入 PDB REMARK 250 记录
│                                             - critic_results["annotated_pdb"] 直接可用
│
├── inverse_optimize/                      ← 逆折叠序列优化
│   ├── __init__.py
│   ├── quality_filter.py                  ← PDB 质量闸门: REMARK iPTM + B-factor pLDDT + CA clash
│   └── mpnn_redesign.py                   ← ProteinMPNN 重设计: 固定界面残基, 采样新序列
│
├── refold/                                ← 重折叠验证
│   ├── __init__.py
│   ├── refolder.py                        ← 底层折叠: refold_distogram() / refold_full()
│   └── screen.py                          ← 两阶段筛选: Scaling(1步,快筛) → Hero(200步,精测)
│
└── bindercraft_filter/                    ← BindCraft 物理评分 (移植自 BindCraft-main)
    ├── __init__.py
    ├── rosetta_scoring.py                 ← Rosetta 界面分析 + FastRelax + RMSD
    ├── structural_analysis.py             ← cKDTree 冲突 + DSSP 二级结构 + 界面残基检测
    ├── filter_check.py                    ← 多维度阈值过滤 (~40项) + failure 统计
    ├── default_filters.json               ← 默认过滤阈值配置
    └── integration.py                     ← ESMFold2 ↔ BindCraft 桥接
```

### 代码拆分设计

为避免本地环境强制依赖 Modal SDK，将原 `run_modal.py` 拆分为三层：

| 文件 | 职责 | Modal 依赖 | 可独立运行 |
|------|------|:----------:|:----------:|
| `core.py` | 全部 Pipeline 逻辑、配置、排序/过滤函数 | ❌ | — (库) |
| `run_modal.py` | Modal 镜像构建、远程部署、`IntegratedPipelineModal` | ✅ | `modal run` / `modal deploy` |
| `run_local.py` | 本地 argparse CLI + `IntegratedPipeline(config).run(...)` | ❌ | `python run_local.py` |

## 模型复用机制

Pipeline 在 Stage 1 和 Stage 4 均需 ESMFold2 模型，但角色不同：

```
Stage 1: ESMFold2Design.load()
  ├─ inversion_models (×2)  → 仅用于 150 步 SGD 设计
  ├─ hf_critic_models (×4)  → Hero Critic 评分 + 后续 Stage 4 复用
  ├─ esmc_model              → 仅用于 pseudoperplexity 正则化
  └─ scaling_critics (×15)  → 逐个加载/评分/删除, 不常驻

Stage 1 完成后:
  → del inversion_models, del esmc_model   ← 回收 ~30GB
  → hf_critic_models 保留, 传入 Stage 4

Stage 4: RefoldScreen
  Scaling 快筛: 加载 1 个轻量 Scaling Critic (~300M base, 1步扩散)
                → 用后立即释放
  Hero   精测: 复用 Stage 1 的多个 Hero Critics
                → 每条序列由所有指定 critic 分别折叠
                → 取 iPTM 最高者为最终结果 (结构 + 指标均来自最优 model)
                → 逐模型加载/释放, 控制显存峰值
```

**与旧版的区别**：

| | 旧版 (重构前) | 当前版本 |
|---|---|---|
| Stage 4 模型来源 | 独立 `ESMFold2Design().load()` | 复用 Stage 1 的 `hf_critic_models` (所有权转移, 逐模型释放) |
| Scaling 模型 | 4× Hero Critic (各 ~3-4 GB) | 1× 轻量 Scaling Critic (~1-2 GB, 用后释放) |
| Hero 模型 | 1× Hero Critic (硬编码 Cutoff2025) | 可配置 N× Hero Critic, 取最优 iPTM |
| Scaling forward/序列 | 4 次 (每 Hero 一次) | 1 次 |
| 额外加载的模型 | inversion×2 + ESMC_lm + hero×4 | 仅 1 个轻量 scaling critic |
| ESMC trunk | 每轮 `load()` 重新加载 | 模块级 `_ESMC_CACHE` 进程内缓存 |

## 梯度体系

所有 ESMFold2 模型权重在加载时冻结 (`_load_hf_model` → `.requires_grad_(False)`)。唯一可训练的是 binder 软序列 logits，梯度通过 `res_type_soft` → Pair Trunk → `distogram_logits` 回传。`DiffusionStructureHead.sample` 带有 `@torch.inference_mode()`，坐标解码端完全不可微。

## 完整数据流

```
Stage 1: ESMFold2 梯度设计 (esmcp)
═══════════════════════════════════════════════════════════════
  target sequence + minibinder(#*N)
    → Inversion Models (150步 SGD, distogram熵 + ESMC正则)
    → 余弦温度退火 (T: 1.0 → 1e-2)
    → Scaling Critics (1步扩散, distogram_iptm_proxy 粗筛)
    → Hero Critics (200步扩散, iPTM + 全原子坐标)
    → 输出: critic_results + esm_design 实例 (模型引用保留)
    → 释放 inversion_models + esmc_model (~30GB)
              │
              ▼
Stage 2: 质量预过滤 (inverse_optimize/quality_filter.py)
═══════════════════════════════════════════════════════════════
  Hero Critic PDB (REMARK 250 含 iPTM/pTM/distogram binding confidence,
                   B-factor 含 pLDDT)
    → 解析 REMARK → iPTM 阈值按 target 链数选择 (单链 0.50 / 多链 0.70)
    → 界面级闸门: distogram binding confidence ≥ 0.50
      (纯 binder×target 界面熵, 无 target-target 稀释)
    → B-factor 均值 (binder 链 CA, 链名从 designed_sequence 推导) → pLDDT ≥ 0.70?
    → cKDTree CA clash → clash == 0?
    → 输出: quality_passed (list[dict])
              │
              ▼
Stage 3: MPNN 逆折叠 (inverse_optimize/mpnn_redesign.py)
═══════════════════════════════════════════════════════════════
  通过的 PDB (target 链在前, binder 为最后一条链)
    → hotspot_residues() 检测界面残基 (cKDTree, 4Å cutoff,
      多链 target 取全部 target 链原子并集)
    → design_chains = "C,A,B" (binder 置首, colabdesign prep_pos 的
      裸数字作用于第一条链, 免跨链 offset 计算)
    → fix_pos = "A,B,234,235,240,..." (target 链整链固定 +
      binder 界面残基裸编号固定)
    → mk_mpnn_model(v_48_020, soluble) 采样 N 条序列 (默认20)
    → 输出: list[{sequence, score, seqid, ...}] (binder 为输出前段)
    → binder_chain/target_chains 从 designed_sequence 自动推导,
      无默认链假设
              │
              ▼
Stage 4: 重折叠筛选 (refold/)
═══════════════════════════════════════════════════════════════
  MPNN 序列 (只有字符串, 无结构坐标)
    → 加载轻量 Scaling Critic (300M base, 用后释放)
    → Scaling 快筛 (1步扩散, distogram_iptm_proxy ≥ 0.50)
    → Hero 精测 (可配置 N 个 Hero Critic, 各 200 步扩散)
    → 每条序列取 iPTM 最高的 critic 的结构 + 指标, 其余丢弃
    → pdb_with_quality_remarks() 嵌入 REMARK 标注
    → 输出: refold_results (list[{annotated_pdb, iptm, hero_model, ...}])
    → 逐模型释放 Hero Critic (~20GB), 为 Rosetta 腾空间
              │
              ▼
Stage 5: Rosetta 物理评分 (bindcraft_filter/), 可选
═══════════════════════════════════════════════════════════════
  Hero 精测 PDB
    → calculate_clash_score()          ← 弛豫前冲突 (链无关)
    → pr_relax()                       ← Rosetta FastRelax (链无关)
    → score_interface()                ← set_interface("AB_C"): target 组
                                         (多链连写) 对 binder; dG, SC,
                                         Hbonds, UnsatHbonds, ...
    → calc_ss_percentage()             ← DSSP 二级结构 (界面残基检测
                                         多链并集)
    → binder_chain/target_chains 从 "target_seq" 推导注入, 无默认链假设
    → 输出: rosetta_screening (dict)
              │
              ▼
Stage 6: 过滤 & 排序 (core.py)
═══════════════════════════════════════════════════════════════
  汇总所有 PipelineResult
    → compute_selection_score()         ← 0.5×norm(iPTM) + 0.5×norm(proxy)
    → check_filters()                   ← BindCraft 二元阈值 (PASSED/FAILED)
    → pI < 6.0                          ← minibinder 等电点筛选
    → select_top_n(N=84)                ← 每 (target, binder) 组择优
    → 保存: selection.parquet + Top-N PDB
```

## 评分体系对照

| 维度 | ESMFold2 原生 | MPNN 逆折叠 | BindCraft 物理 |
|------|:-----------:|:---------:|:------------:|
| 界面置信度 | iPTM | — | — |
| 全局折叠置信度 | pTM | — | — |
| 局部置信度 | pLDDT (B-factor) | — | — |
| Distogram 结合 | distogram_iptm_proxy | — | — |
| 序列多样性 | — | MPNN sampling | — |
| 界面能量 | — | — | dG, dSASA |
| 形状互补 | — | — | ShapeComplementarity |
| 氢键 | — | — | InterfaceHbonds, UnsatHbonds |
| 原子冲突 | — | — | cKDTree clash |
| 二级结构 | — | — | DSSP Helix/Sheet/Loop% |
| 等电点 | pI < 6.0 | — | — |

## 链名约定

所有模块统一使用以下约定 (与 AlphaFold 惯例一致)。复合物链名由 `asym_id` 顺序决定 (target 链在前, binder 恒为最后一条链)：

| 链 ID | 含义 | 来源 |
|--------|------|------|
| A | Target 链 1 | `_asym_id_to_chain_label(0)` → `"A"` |
| B | Target 链 2 (多链 target 时) | `_asym_id_to_chain_label(1)` → `"B"` |
| C, D... | 后续 target 链 | `_asym_id_to_chain_label(2+)` |
| 最后一条链 | Binder | `_asym_id_to_chain_label(len(target_chains))` |

**注意**: 多链 target 时 binder 链名**不再固定为 "B"** (如两条 target 链时 binder = 链 "C")。`PipelineResult` 携带 `target_chains` / `chain_labels` / `binder_label` 字段供下游阶段使用。

### Hotspot 格式 (链限定)

Hotspot ID 必须带链前缀: `"<chain>:<residue><idx>"`，如 `"A:L150"` = 链 A 的第 150 位 (1-indexed, 链内编号)。链标签限定 A-Z, 只能标注 target 链, 不能标注 binder 链。

esmcp 输出的 `annotated_pdb` 中 B-factor = pLDDT × 100, REMARK 250 记录 iPTM/pTM/distogram_binding_confidence。

## 快速使用

### 本地运行 (推荐, 无需 Modal)

```bash
# 基础: ESMFold2 设计 + 质量过滤
python run_local.py --target-name pd-l1 --binder-name minibinder

# 完整: ESMFold2 设计 + MPNN 逆折叠 + 重折叠
python run_local.py --target-name pd-l1 --binder-name minibinder \
    --enable-mpnn --mpnn-num-seqs 20 --top-n 20

# 抗体设计 + hotspot 引导 (链限定格式 <chain>:<residue><idx>)
python run_local.py --target-name cd45 --binder-name trastuzumab_framework_vhvl \
    --enable-mpnn --target-hotspot-ids A:L150 A:Y123

# 多链 target + 多链 hotspot (binder 自动为最后一条链)
python run_local.py --target-name my_complex --binder-name minibinder \
    --target-sequence "SEQ1|SEQ2" --enable-mpnn \
    --target-hotspot-ids A:K32 B:D45

# 多模型 Hero Critic 重折叠 (取最优 iPTM)
python run_local.py --target-name pd-l1 --binder-name minibinder \
    --enable-mpnn --mpnn-num-seqs 30 \
    --hero-critics ESMFold2-Experimental-Cutoff2025 ESMFold2-Experimental

# 自定义序列
python run_local.py --target-name my_target --binder-name my_binder \
    --target-sequence "MQIFVK..." --binder-sequence "##########" \
    --enable-mpnn
```

### Modal 远程运行

```bash
# 部署 (首次)
modal deploy run_modal.py

# 单任务
modal run run_modal.py --target-name pd-l1 --binder-name minibinder --enable-mpnn

# 完整流程 (含 Rosetta, 需 PyRosetta 许可)
modal run run_modal.py --target-name egfr --binder-name minibinder \
    --enable-mpnn --enable-rosetta --mpnn-num-seqs 30
```

### Python API

```python
# 本地: 直接导入 core 模块
from core import IntegratedPipeline, PipelineConfig

config = PipelineConfig(
    enable_mpnn=True, mpnn_num_seqs=20, top_n=20,
    hero_critic_names=["ESMFold2-Experimental-Cutoff2025", "ESMFold2-Experimental"],
)
pipeline = IntegratedPipeline(config)
result = pipeline.run(target_name="pd-l1", binder_name="minibinder", seed=42)

for r in result["selected"][:5]:
    print(f"{r.selection_score:.4f}  {r.binder_sequence[:30]}...")
```

```python
# 远程: Modal API
import modal

IntegratedPipelineModal = modal.Cls.from_name(
    "integrated-binder-pipeline", "IntegratedPipelineModal"
)
pipeline = IntegratedPipelineModal()
result = pipeline.run.remote(target_name="pd-l1", binder_name="minibinder", seed=42)
```

```python
# 分步使用 (各模块独立调用)
from inverse_optimize import filter_pdb_by_quality, mpnn_redesign_binder
from refold import RefoldScreen, RefoldScreenConfig

# 质量检查
qc = filter_pdb_by_quality("design.pdb", min_iptm=0.7, binder_chain="B")

# MPNN
designs = mpnn_redesign_binder("design.pdb", num_seqs=20)

# 重折叠
screen = RefoldScreen(target_seq, scaling_models, hero_models)
hero_results, _ = screen.run([d["sequence"] for d in designs])
```

## 外部依赖

### 核心依赖 (core.py / run_local.py)

```
  - torch ≥ 2.8.0 + triton + flash-attn
  - transformers (HuggingFace, ESMFold2 fork)
  - esm (Biohub GitHub @main)
  - biotite, abnumber (抗体时)
  - biopython, scipy, numpy, pandas, pyarrow
  - colabdesign (ProteinMPNN)
```

### Modal 专用 (run_modal.py)

```
  - modal (远程 GPU 编排)
  - CUDA 12.8 + transformer-engine + xformers (由 Modal 镜像提供)
```

### Rosetta 物理评分 (可选)

```
  - pyrosetta (商业许可)
  - DSSP 可执行文件 (mkdssp 或 dssp)
  - DAlphaBall.gcc (Rosetta UnsatHbonds 检测)
```

## 原始文件来源

| 本模块文件 | 原始来源 |
|-----------|---------|
| `esmcp/binder_design.py` | `esmfold2/binder_design/binder_design.py` (新增 REMARK + pdb_with_quality_remarks) |
| `core.py` | `run_modal.py` 核心逻辑抽取 (2026-08 拆分) |
| `run_modal.py` | Modal 胶水提取自原 `run_modal.py` |
| `run_local.py` | 新建本地入口 (2026-08) |
| `rosetta_scoring.py` | `BindCraft-main/functions/pyrosetta_utils.py:1-243` |
| `structural_analysis.py` | `BindCraft-main/functions/biopython_utils.py:1-236` |
| `filter_check.py` | `BindCraft-main/functions/generic_utils.py:56-411` |
| `default_filters.json` | `BindCraft-main/settings_filters/default_filters.json` |

## 优化方向

以下为已知局限和未来改进方向：

### 1. 多链 Target 支持 (全部 Stage 已完成)

**已完成 (2026/8/13)**：
- `design_binder()` 移除 target 单链 assert；`build_target_one_hot()` 逐链编码 target one-hot (链独立, 按 token 序拼接)
- `fold_and_get_distogram()` 修复同序列链被 dict 去重静默丢弃的 bug (同源多聚体 target)
- hotspot 改为链限定格式 `"A:L150"` (方案 B)，`get_epitope_mask()` 按链解析并校验
- `PipelineResult` 新增 `target_chains` / `chain_labels` / `binder_label` 链元数据
- `core.py run()` 的 `target_seq` 推导修复 (保留除 binder 外全部链)
- 设计损失 (`intra_contact` / `inter_contact` / `glob`) 与 `compute_distogram_iptm_proxy` 为位置逻辑, 多链天然兼容
- Stage 2: `quality_filter` 的 binder 链名必填 (无默认 "B"), 从 `designed_sequence` 推导注入; iPTM 双阈值 (单链 0.50 / 多链 0.70); 新增界面级 distogram binding confidence 闸门 (修复 score_critic 的 REMARK key 映射 bug)
- Stage 3: MPNN `design_chains` binder 置首 ("C,A,B") 绕开 colabdesign fix_pos 跨链 offset; `hotspot_residues` 多链 target 原子并集; 链名从 `designed_sequence` 自动推导
- Stage 4: `refold_distogram` 使用 `build_target_one_hot` (逐链编码), 位置逻辑天然多链
- Stage 5: Rosetta `set_interface("AB_C")` target 组对 binder; `score_interface`/`calc_ss_percentage` 多链并集界面检测; 链名从 `target_seq` 推导注入

**待改造 (下游阶段)**：无 — Stage 2-5 已全部完成 (2026/8/13)。

### 2. ProteinMPNN 权重与噪声策略

当前 MPNN 使用 `backbone_noise=0.00`, `weights="soluble"`，完全信任 ESMFold2 Hero Critic 的扩散 backbone。ESMFold2 的扩散解码器坐标精度低于 AF2 的 IPA 迭代，适当增加 `backbone_noise`（如 0.02）可提升序列多样性。此外可对比 `vanilla` / `noise` 等 MPNN 权重对最终设计质量的影响。

### 3. 多模型 Hero Critic 重折叠 (已部分实现)

`PipelineConfig.hero_critic_names` 允许指定多个 Hero Critic, 每条 MPNN 序列取 iPTM 最高模型的输出。当前为 "winner-takes-all" 策略, 不取均值。如需 ensemble 平均 iPTM/pTM (类似 BindCraft 的 5 模型平均), 可扩展 `RefoldScreen.hero_predict()` 增加 `ensemble_mode: "best" | "average"` 选项。

### 4. Rosetta 评分融入综合排名

当前 `selection_score` 默认仅使用 iPTM + distogram_iptm_proxy。当 `enable_rosetta=True` 时可通过 `rosetta_weight` 引入物理评分（dG, SC, Hbonds），但权重和各子项系数需要针对具体靶点校准。

### 5. AF2 验证替代 ESMFold2 重折叠

当前 MPNN 序列用 ESMFold2 重折叠验证。由于 MPNN 基于 PDB 晶体结构训练（与 AF2 的物理世界更一致），可尝试用 ColabDesign AF2 前向折叠验证 MPNN 序列，并与 ESMFold2 验证结果做交叉对比。

### 6. CDR 特异的抗体评分

当前 `is_antibody` 参数已传递至各模块但未在 `_collect_results()` 中使用。notebook 中抗体设计使用 `cdr_distogram_iptm_proxy` 替代 `distogram_iptm_proxy` 进行排序。需在 `compute_selection_score()` 中根据 `is_antibody` 切换评分来源。

### 7. Scaling Critic 模型选择可配置化

当前 Stage 4 使用硬编码的 `ESMFold2-Experimental-Fast-base300M-step1500k` 作为 scaling critic。不同靶点/场景可能受益于不同规模（300M/600M/6B）或不同训练步数（250k/500k/.../1500k）的 checkpoint。可将其暴露为 `PipelineConfig.scaling_critic_name` 选项（Hero Critic 已通过 `PipelineConfig.hero_critic_names` 支持多模型选择）。
