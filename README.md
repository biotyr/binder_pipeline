# ESMFold2 × ProteinMPNN × BindCraft 集成 Pipeline

端到端 Binder 设计流程：ESMFold2 梯度设计 → 逆折叠优化 → 重折叠验证 → 物理化学过滤。

## 目录结构

```
pipeline/
├── README.md                              ← 本文件
│
├── environment.yml                        ← conda 环境配置 (esmbind, python 3.12,
│                                            含版本裁决依据与手动安装步骤)
│
├── core.py                                ← 核心模块 (纯 Python, 零 Modal 依赖)
│   ├── PipelineConfig / PipelineResult
│   ├── compute_selection_score / apply_filters / select_top_n
│   ├── IntegratedPipeline (6 阶段 run())
│   ├── _hotspot_contact_check (热点几何核验门, 热点任务 Stage-2 附加闸)
│   └── _collect_results / _save_outputs
│
├── run_modal.py                           ← Modal 远程部署适配器
│   └── _get_modal_image() / IntegratedPipelineModal / main()
│
├── run_local.py                           ← 纯本地运行入口
│   └── argparse → IntegratedPipeline(config).run(...)
│
├── esmcp/                                 ← ESMFold2 设计引擎
│   ├── binder_design.py                   ← 核心: 150步 SGD + Inversion/Critic 评分
│   │                                        - pdb_with_quality_remarks() 将 iPTM/pTM
│   │                                          嵌入 PDB REMARK 250 记录
│   │                                        - critic_results["annotated_pdb"] 直接可用
│   │                                        - inter loss 双分支 (full_coverage /
│   │                                          hotspot_neighborhood) + fold-rescue
│   │                                          三阶段调度 (见「热点引导设计」)
│   └── binder_design.ipynb                ← 上游 Modal 教程 notebook (sweep → 收集
│                                            → 选序参考实现)
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

另: `backup/` 为多链 target 改造 (2026/8/13) 前的单链版本快照, 见 `backup/read.md`。

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
  └─ scaling_critics (×15)  → pipeline 默认关闭 (core.py 传
                              use_scaling_critics=False: base6B 组各自整载
                              6B 主干, 瞬态峰 ~+25GB/critic, 4090 不可容),
                              仅 H100 诊断路径; 启用时逐个加载/评分/删除

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
    → Inversion Models (150步 SGD, distogram熵 + ESMC正则; 热点任务 inter
      loss 默认走热点邻域分支, 可选 fold-rescue 调度, 见「热点引导设计」)
    → 余弦温度退火 (T: 1.0 → 1e-2)
    → Scaling Critics (1步扩散, distogram_iptm_proxy 粗筛; pipeline 默认关闭)
    → Hero Critics (200步扩散, iPTM + 全原子坐标)
    → 输出: critic_results + esm_design 实例 (模型引用保留)
    → 释放 inversion_models + esmc_model (~30GB)
              │
              ▼
Stage 2: 质量预过滤 (inverse_optimize/quality_filter.py)
═══════════════════════════════════════════════════════════════
  Hero Critic PDB (REMARK 250 含 iPTM/pTM/distogram binding confidence,
                   B-factor 含 pLDDT)
    → 解析 REMARK → 界面置信度双门: pair_chains_iptm 统一 0.3 (粗筛)
      + ipsae 0.5 (补充门, 2026/10/19 定标)
    → 界面级闸门: distogram binding confidence ≥ 0.50
      (纯 binder×target 界面熵, 无 target-target 稀释)
    → B-factor 均值 (binder 链 CA, 链名从 designed_sequence 推导) → pLDDT ≥ 0.70?
    → cKDTree CA clash → clash == 0?
    → 热点任务追加几何核验: 每热点 min binder Cβ ≤10Å ∧ 12Å 内 binder
      残基 ≥3 (质量门不度量热点 honoring, 2026/10/14 新增)
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
    → compute_selection_score()         ← 0.5×norm(界面置信度) + 0.5×norm(proxy)
                                          (ipTM 项默认 ipSAE; Average_i_pTM
                                          过滤槽固定 pair_chains_iptm 值)
    → check_filters()                   ← BindCraft 二元阈值 (PASSED/FAILED)
    → pI < 6.0                          ← minibinder 等电点筛选
    → select_top_n(N=84)                ← 每 (target, binder) 组择优
    → 保存: selection.parquet + Top-N PDB
```

## 评分体系对照

| 维度 | ESMFold2 原生 | MPNN 逆折叠 | BindCraft 物理 |
|------|:-----------:|:---------:|:------------:|
| 界面置信度 | pair_chains_iptm (默认; 全局 iPTM 备选) | — | — |
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

### 输入链名与 hotspot 的同步 (链名重置)

Pipeline 本体不解析 FASTA: `run()` 接受 `|` 拼接的序列串, 原始链名**不保留** —— 复合物内一律按 asym_id 位置重置 (见上表)。因此 hotspot 的链前缀指向的是**传入序列中该链的排列位置**, 而非 FASTA/PDB 原始链字母:

- 单链 target `"SEQ1"` → 重置为 `A`; hotspot 写 `A:<res><idx>`
- 多链 target `"SEQ1|SEQ2"` → 依次重置为 `A`、`B` (binder 为 `C`); hotspot `B:P98` 指**第二段** (SEQ2) 链内第 98 位 (1-indexed)

对齐由调用方保证: 原始链名 → 重置链名的映射 = 传入时链的排列顺序。`test/e2e/run_e2e.py` 按 FASTA header 的 Chain 字母经 `chain_order` **显式排序**后拼接, 因此两个测试靶 (3RRQ 单链 A; 4ZQK 双链 A=PD-L1, B=PD-1) 重置前后链名一致, 4/4 热点残基字母核对通过。注意 4ZQK FASTA 中 B 链记录排在 A 链之前 —— 若按文件顺序拼接会使 B→A 错位, `chain_order` 显式排序即为规避。

两个校验护栏: ① hotspot 残基字母必须与该位置序列字母一致, 错位时立即暴露; ② 残基编号是去 X 后序列的链内 1-indexed 位置。

## 热点引导设计 (2026-10 源码化)

背景: 双链靶上 inter contact loss 的"全覆盖"语义 (对全部 target 残基平均)
与 epitope loss 的"局域钉扎"存在结构性冲突 (4ZQK 实证: 交界边缘热点 w0.5 →
退化解; 非界面热点 → 弃热点保界面; 单链靶 3RRQ 无此冲突)。2026-10-12/14
以下三项入正式源码; **不指定热点时默认行为逐位不变**。

### 1. inter loss 双分支 — `inter_loss_mode: str | None = None` (run/design 参数)

- `None` (默认 auto): 指定热点 → `"hotspot_neighborhood"`, 无热点 →
  `"full_coverage"`。**语义变更 (2026-10-14 拍板): 热点任务自此默认走邻域
  分支** (无论入口是否显式传参)。
- `"hotspot_neighborhood"`: inter loss 的平均域从全部 target 残基收窄为
  热点邻域可及残基。mask 在优化前由一次**纯 target 折叠** (无占位 binder,
  mask 是 target 的纯函数, 不随 binder 长度/组成波动) 的 distogram 计算:
  近热点 (到任一热点的期望距离 < 15Å) ∧ 可及 (10Å 接触数 burial proxy
  低于 target 内 q0.5 分位, |i-j|<2 的配对不计), 强制含热点; 存活 < 10
  残基时回退全覆盖并告警。loss 引擎与 cutoffs 与全覆盖一致 (k=1, 22Å)。
  晶体校准 (4ZQK): mask vs 晶体接触数 Spearman 0.932 / vs DSSP RSA
  −0.836; R=10-12Å 与晶体真值一致性最高。
- `"full_coverage"`: 原路径, 对热点任务显式回退旧行为。

阈值常量 `INTER_*` (esmcp/binder_design.py) 为 provisional defaults。

### 2. fold-rescue 三阶段权重调度 — `fold_rescue_schedule: bool | None = None`

**语义变更 (2026-10-21 拍板, 与 ① 的 inter_loss_mode 同款 auto 约定)**:
`None` (默认) 自动解析 — **热点任务默认启用三阶段调度**, 无热点任务保持
关闭; 显式 `False` 可退出 (旧行为), 无热点时显式 `True` 仍 ValueError。
调度内容: P1 原生权重起步; 步 25/35 检查点发现 intra last5 均值 > 3.8
(折叠横盘) → P2 折叠挽救 (inter/epitope 降至 0.1/0.1); 步 ≥50 起每 5 步
检查, intra last10 均值 < 3.0 → P3 界面精修 (恢复 0.5/0.5)。相位转移只改
损失权重, 其余一切不动; 事件流写入 `critic_results["fold_rescue_events"]`。
参数常量 `FOLD_RESCUE_*`。

### 3. 热点几何核验门 — `IntegratedPipeline._hotspot_contact_check`

质量门 (iPTM/pLDDT/proxy) 不度量热点 honoring —— iPTM 0.78 的设计可对热点
28-30Å 零接触而全线过门 (多 seed 实证)。指定热点且 `run()` 传入
`target_sequence` 时自动启用: hero 结构中**每个**热点要求最近 binder Cβ
(gly 用 CA) ≤10Å 且 12Å 内 binder 残基 ≥3; 核验失败进 `quality_failed`,
条目携带 `hotspot_check` detail。PDB 为复合体顺序编号, 热点按链内序号 +
前序靶链长度换算定位, 残基字母不符时 ±2 容差找回。注意: preset target 名
(不传序列) 时该门不生效。

### 入口

| 入口 | 新参数 |
|------|--------|
| Python API | `IntegratedPipeline.run(..., inter_loss_mode=..., fold_rescue_schedule=...)` (均 `None`=auto) |
| e2e runner | `test/e2e/run_e2e.py --inter-loss-mode {auto,full_coverage,hotspot_neighborhood}` (默认 auto) + `--fold-rescue-schedule {auto,on,off}` (默认 auto = 热点任务启用) |
| `run_local.py` / `run_modal.py` | 未暴露新参数; 热点任务自动走 ①② 默认, 需显式控制时用 Python API |

### 4. 策略选择经验法则：inter 域随 binder/target 长度需求比（2026-10-27 双臂 benchmark 实证）

2026-10-23/27 两轮双臂 benchmark（20 靶 × 2 策略 × 3 seed = 60 组有效配对，ipSAE
口径；完整数据与报告见 `benchmark/results/report_ipsae_length.md`）发现：
**① 邻域化与 ② fold-rescue 调度哪个占优，由需求比 = target 残基数 / binder 长度
（全覆盖损失的"单位 binder 容量接触需求"）决定，临界线在需求比 ≈1**，而非靶长
绝对值。三段经验法则：

| 需求比 | 机制 | 实证（均值 ΔipSAE = current−native） | 建议 |
|---|---|---|---|
| **≥2**（binder ≪ target） | 全覆盖物理不可满足，inter 退化为稀释噪声并与 epitope 拔河 | **current 占优 4/4 靶**（GDF-8 +0.26 / BHRF1 +0.13 / IL-7Rα +0.12 / 15-PGDH +0.05） | **用 current（①② 默认即最优）** |
| **1.0-1.8**（binder 略小于 target，全覆盖"临界可达"） | 稠密全覆盖梯度是有效塑形压力；邻域化撤压反而受损 | native 7/7 靶均 ≥ current（TrkA −0.25 最显著）；但有 Der-f7 型 native 三 seed 坍塌风险 | **倾向 native（显式 `inter_loss_mode="full_coverage"`），建议双跑** |
| **<1**（binder ≥ target，全覆盖近乎免费） | 两种 inter 域趋同，epitope 被子集化，方差主导 | 两轮证据矛盾（长 binder 批 current +0.15，L120 同靶判决翻负） | **双跑取优** |

**判决实验**（U 型模型的最强证据）：同靶唯一变量 binder 长度（181-196 → 120）
跨过需求比 1.0 边界——IL-17A 的 Δ 从 +0.165 翻到 **−0.231**、Der-f21 从 +0.125
翻到 **−0.244**（6 个 seed 配对 5 个负向），需求比 <1 时 current 占优、跨入
1.0-1.8 后优势消失，方向与模型一致。

**限制与注意**：
- 样本 = 15 靶 / 3 seed，单指标（ipSAE）；长 binder（>180aa）区 seed 方差
  ±0.4-0.6，单 seed 配对差异不可作结论。
- 中带并非 native 稳赢：Der-f7 在需求比 1.63 下 native 三 seed 全坍塌
  （ipSAE ≤0.1）——临界带存在 native 自身失效模式，故建议双跑。
- 策略差异与热点 honoring 相互独立：高 ipSAE 不保证热点接触
  （IL-17A 长 binder 例），热点任务的几何核验门在任何策略下都必须保留。
- `inter_loss_mode` 默认（`None` → 邻域）在需求比 ≥2 区间最优；临界带任务
  请按上表显式选边或双跑。

## 置信度指标: binder-target pair ipTM (2026-10-19 切换默认)

Stage-2 质量门与 Stage-6 过滤/排序的界面置信度槽位已从全局 ipTM 切换为
**pair_chains_iptm** —— ESMFold2 confidence head 的链对子块 TM 均值
(`pair_chains_iptm[target][binder]`), 多链 target 按**残基加权合并**:

```
merged = Σᵢ Lᵢ·pair[tᵢ][binder] / Σᵢ Lᵢ   ≡  binder×(全部 target) 子块均值
```

- **方向 t2b**: PAE 语义为"行=对齐参考、列=误差对象", `[target][binder]`
  即**固定 target 正确、评估 binder 的位置置信度** —— binder 质量的定义。
  反方向 b2t 仅写入 REMARK/结果 dict 作诊断, 不进任何消费路径。
- **修复的缺陷**: 全局 ipTM 混入 target-target 链间置信度 (native 界面
  预测极好 → 虚高)。实证: 退化"包裹型"设计全局 iPTM 0.888 (全场第二),
  pair 值仅 0.255 —— 膨胀被精确剥离。
- **语义边界**: 这是**组装置信度** (复合体是否良定), 不是接触度量 ——
  未结合链经 target 内部界面传递置信度仍中高, 属合法贡献。接触意图
  (热点 honoring) 由热点几何核验门另行承担, 两门各司其职。
- **定标** (diag_pair_iptm_replay, 8 真实产物 hero 200 步重折):
  真解多链 0.487-0.684 / 真解单链 0.563-0.586 / 退化样本 ≤0.255 →
  阈值 `quality_min_iptm_multi` 0.70→**0.40** (分离区间 (0.26, 0.49)
  取中), `quality_min_iptm_single` 0.50 (沿用, 真解留 0.06 边际),
  `Average_i_pTM` 槽 0.5→**0.4**。块均值系统性低于旧全局值 (-0.2~-0.3),
  新旧数值不可混比。
- **门与排序解耦 (2026-10-19 定稿)**: Stage-2 质量门固定 pair 门
  (统一 0.3 粗筛 —— pair 值的 d0 耦合全复合体长度, 跨设计尺度不稳,
  精细鉴别交给 ipSAE 门) + ipSAE 补充门 (0.5) 双门, 与排序开关无关;
  `Average_i_pTM` 过滤槽固定 pair 值。`PipelineConfig.iptm_metric` 仅
  控制 Stage-6 selection score 的 ipTM 项与 hero 比较 (默认 "ipsae",
  见下节); e2e/run_local 经 `--iptm-metric {ipsae,pair_chains,global}`。
  REMARK 250 键: `pair_chains_iptm` (被门消费) / `pair_chains_iptm_b2t`
  (诊断) / `pair_chains_iptm_chains` (逐链人类可读)。

### ipSAE 补充信号 (2026-10-19)

`ipsae.py` (pipeline 根目录, 与 core.py 同级 —— 所有 hero 精确折叠处共用):
参考 Dunbrack lab ipsae.py v4 的**设 cutoff 主分数 (d0res)** 自实现, 纯
numpy; 等价性测试对参考脚本 Example 数据 6 链对全部 <1e-6
(`test_ipsae.py`)。方向 t2b, 与 pair_chains_iptm 同位提取 (hero fold
`output["pae"]` → 同源同折), REMARK 键 `ipsae` (可解析) / `ipsae_b2t` /
`ipsae_anchor` (锚残基 flat idx —— 界面定位诊断)。
**算术域 = 仅 binder↔target 对**: 列掩码严格限定 binder, target 内部
链对不进 n0res 也不进均值 (不变性单测钉死) —— 多链 target 的
target-target 置信度不会混入, 与 pair_chains_iptm 同语义。

- **补充角色**: Stage-2 `min_ipsae` 补充门 (默认 **0.5**, 与排序开关无关
  常透传) + Stage-6 selection score ipTM 项**默认源**
  (`iptm_metric="ipsae"`, 2026-10-19 定稿)。
- **定标 (8 产物, cutoff15)**: 全流程验证真解 (3 个 selected 设计)
  **0.797-0.813**; 未验证/带缺陷产物 0.15-0.38 (founder 0.152 —— pair 值
  0.494 过粗筛门但 Stage-6 pLDDT FAIL + 序列组成单调, 从非验证成功;
  seed3 0.254 热点假阳性; wrapper 0.379 低复杂度); 灾难失败 0.00-0.02。
  **ipSAE 排序与验证状态一致**, 恰好补充 pair 门盲区: 拦截 founder 型
  "pair 值过门但整体质量缺陷" 设计 (门 0.5 位于分离带 (0.38, 0.80),
  真解留 ~0.3 边际; n=3 真解样本, 待更多 e2e 复核)。
- **注意事项**: d0res 的 d0 由锚残基低 PAE 伙伴数决定, 对界面大小敏感
  (窄界面分数系统性偏低) —— 阈值语义是"已验证设计 ≈0.8"的经验锚定,
  换 binder 类型/长度需重新校核; cutoff 15 与 10 排序一致。
- **与 pair 值数值不同族** (点值 PAE + 配对专属 d0 vs soft-bin 期望 +
  全复合体 d0), 两者是同级补充信号, 各走各的定标, 不可互换。

## 快速使用

### 本地运行 (推荐, 无需 Modal 远程服务; 仍需安装 modal 包, 见外部依赖)

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

# 热点引导: 热点任务自动走邻域 inter 分支 + fold-rescue 三阶段调度 +
# Stage-2 热点几何门 (需传序列)
result = pipeline.run(
    target_name="my_complex", binder_name="minibinder", seed=42,
    target_sequence="SEQ1|SEQ2",
    target_hotspot_ids=["A:K32", "B:D45"],
    # inter_loss_mode="full_coverage",   # 显式回退全覆盖旧行为
    # fold_rescue_schedule=False,       # 显式关闭三阶段调度 (热点任务默认开)
)
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

# MPNN (binder_chain / target_chains 必填, 无默认链假设)
designs = mpnn_redesign_binder(
    "design.pdb", binder_chain="B", target_chains="A", num_seqs=20,
)

# 重折叠
screen = RefoldScreen(target_seq, scaling_models, hero_models)
hero_results, _ = screen.run([d["sequence"] for d in designs])
```

## 外部依赖

完整环境规格以 `environment.yml` 为准 (conda env `esmbind`, python 3.12,
含版本裁决依据与手动安装步骤)。核心依赖:

```
  - torch==2.8.0 + triton==3.4.0 + flash-attn==2.8.3 (cu128 预编译 wheel)
  - transformers — Biohub fork (github.com/Biohub/transformers), 提供
    mainline 没有的 transformers.models.esmfold2_* 模块
  - esm 3.3.0 — Biohub fork 本地快照 (/home/taoyr/projects/biohub-esm/
    esm-main; 上游 git@main 已漂移到 3.4.1, 要求 torch>=2.11,
    与 torch==2.8.0 冲突, 故钉本地克隆)
  - biotite, abnumber (抗体时; 依赖 conda 渠道的 anarci + hmmer)
  - biopython, scipy, numpy, pandas, pyarrow
  - colabdesign (ProteinMPNN; 依赖 conda 渠道 JAX 栈 jax/jaxlib/flax/
    dm-haiku 等, 安装须 --no-deps, 见 environment.yml 尾部说明)
  - modal — esmcp/binder_design.py 模块级 import, 本地运行同样需要安装
```

### Modal 专用 (run_modal.py)

```
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

### 8. 热点分支阈值校准与多 seed 复核

`INTER_*` 邻域阈值与 `FOLD_RESCUE_*` 调度参数按 4ZQK 校准 (provisional defaults), 换靶需复核。Stage-1 单轮对比受运行彩票支配 (同配置同 seed 两次运行 iPTM 相差可达 ±0.09, inductor autotune 按计时选 kernel → bf16 数值微差被大步长 SGD 放大), 热点任务结论应多 seed 验证; 过程量 (proxy loss 轨迹) 与真实质量可脱钩, 以 hero 事后判别 (iPTM/pLDDT/热点几何) 为准。
