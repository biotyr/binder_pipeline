# Pipeline 模块测试报告

**日期**: 2026-09-09  **环境**: esmbind (python 3.12.14 / torch 2.8.0+cu128)
**范围**: Stage 2–6 模块级测试。Stage 1 (ESMFold2 设计) 需 ~33GB fp32 显存,
超出单张 4090, 留待端到端阶段 (见文末"遗留事项")。

## 一、Stage 1 输出模拟方案

Stage 1 的输出是一组 `critic_results` dict (annotated_pdb / designed_sequence /
complex), 不是文件。模拟器 `test/module_tests/build_stage1_sim.py` 用真实晶体
结构构造:

| 场景 | 构成 | 来源 |
|---|---|---|
| 单链 (3RRQ 任务) | target=PD-1(链A) + 伪binder=PD-L1(链B) | 4ZQK 链重排 (真实界面) |
| 多链 (4ZQK 任务) | target=PD-L1(A)+PD-1(B) 双链 + 伪binder=PD-1(链C) | 4ZQK + 3RRQ PD-1 叠合 (RMSD 1.52Å, 游离/结合态正常构象差) 后平移摆位搜索 |

- 质量指标经 pipeline 自己的 `pdb_with_quality_remarks()` 注入 REMARK 250,
  B-factor 列刷成受控 pLDDT → 精确构造过/不过边界样本 (共 7 个, 见
  `sim_pdbs/` + `sim_manifest.json`)。
- 多链摆位约束: 重原子最近距 2.68Å (天然界面范围), 12 个接触残基, 零 Cα 冲突。
- 热点轴: 热点唯一消费点是 Stage 1 epitope loss, 已由 T6 用真实 distogram
  验证数值路径; hotspot.txt 本身已通过 `get_epitope_mask` 官方校验 (4/4)。

## 二、测试结果一览 (9/9 通过)

| 测试 | 内容 | 结果 | 产物 |
|---|---|---|---|
| T1a/T1b | Stage 6 评分/过滤/Top-N + Rosetta 加权分支 | **12/12 checks** | t1_stage6.json |
| T2 | Stage 2 质量闸门 7 个边界样本 | **7/7** (单/多链 iPTM 0.50/0.70、pLDDT 0.70、proxy 0.50、clash=0) | t2_quality_gate.json |
| T3 | Stage 3 MPNN 单链 (20 条, 5s) | len=115 ✓ 无Cys ✓ 界面17残基固定 | t3_t4_mpnn.json |
| T4 | Stage 3 MPNN 多链 (20 条, 5s) | fix_pos=`A,B,33,...` 双链字母+裸数字 ✓ | t3_t4_mpnn.json |
| T5 | Stage 4 scaling 快筛 (300M critic, 1.33GiB) | 8/8 通过 0.50 阈值 (0.879–0.897) | t5_t6_scaling.json |
| T6 | epitope-loss 真实 distogram 数值路径 | 有热点→有限 loss (8.85/2.34); 无热点→无该项 | t5_t6_scaling.json |
| T7a/b/c | Stage 5 Rosetta 评分 (含 FastRelax 156s) | 界面残基数与 T3/T4 交叉一致; relax 后 dG +92→−59.6, hbonds 6→14 | t7_rosetta.json |

亮点验证:
- **链名推导一致性**: core.py `_derive_chain_metadata` 与所有模拟样本一致
  (多链 binder=C ≠ B 的关键路径)。
- **T7 界面残基数交叉验证**: Rosetta 检出 17/12 个界面残基, 与 MPNN 固定数
  (17) 和摆位接触数 (12) 完全一致 — 三个独立实现的界面检测互相印证。
- **边界含等号语义**: iPTM=0.50 恰好通过 (`<阈值` 才拒绝), 与 BindCraft 一致。

## 三、发现并修复的 4 个 pipeline bug (均经用户确认)

| # | 文件 | 问题 | 触发条件 | 严重性 |
|---|---|---|---|---|
| 1 | `inverse_optimize/mpnn_redesign.py:292,306,307` | `if raw.get("seq")` 对 colabdesign 返回的 numpy 数组 (≥2元素) 抛歧义真值异常 | Stage 3 默认 mpnn_num_seqs=20 **必崩** | 崩溃 |
| 2 | 同文件 `:292-310` | binder 长度被赋值为全链拼接总长; 且 colabdesign 用 `/` 分隔多链序列, 旧前缀切片恒不生效 → "binder 序列"实为 binder+target 拼接 | Stage 3 **静默产出垃圾序列**, Stage 4 把 target 当 binder 二次折叠 | 数据损坏 |
| 3 | `esmcp/binder_design.py:554` | 热点分支引用未定义的 `target_length` (重构残留, 正确值 `total_target` 在 538 行) | 任何**带热点**的设计第 0 步 NameError | 崩溃 |
| 4 | `bindercraft_filter/rosetta_scoring.py:97` | PyRosetta 2026.29 移除 `set_interface(str)` 重载, 要求 `DockingPartners` 对象 | Stage 5 必崩 | 崩溃 |

修复: #1/#2 显式 None+长度检查、按 `/` 分段取 binder (binder-first 设计);
#3 `target_length=total_target`; #4 try/except 兼容 shim
(旧版字符串路径不变, 新版走 `docking_partners_from_string`)。
**注意**: bug #2/#3 意味着修复前默认端到端流程无法产出任何正确结果
(bug #2 静默、bug #3 崩溃)。

## 四、遗留事项 (端到端 4 任务待办)

1. **显存**: Stage 1 fp32 需 ~33GB (ESMC-6B 主干 23.7GB + 6 模型头 + 激活),
   超出 4090 24GB。待有空卡后需决策 bf16 主干 (~17-20GB, 数值轻微偏差) 或
   其他方案。
2. **运行时注意**:
   - `HF_HUB_OFFLINE=1` (huggingface.co 被墙, 权重已全量本地缓存);
   - 端到端配置需显式设 `rosetta_dssp_path`(本机 dssp 二进制在
     `esmbind/bin/dssp`, Biopython DSSP 默认找 `mkdssp`, 传空串会失败);
   - `pr.init` 需带 `-holes:dalphaball ...` (BindCraft 原版行为; T7 已验证),
     端到端脚本启动早期调用一次。
3. **结果目录规范** (用户指定): `test/results/{任务}/{with,no}_hotspot/
   {stage1_only,full_pipeline}/`, 同时保存 Stage1-only 与全流程结果供对比,
   启用 PyRosetta。模块测试产物一律在 `test/module_results/`。

## 五、产物清单

```
test/module_tests/
  build_stage1_sim.py      run_t1_stage6.py     run_t2_quality_gate.py
  run_t3_t4_mpnn.py        run_t5_t6_scaling.py run_t7_rosetta.py
  sim_manifest.json        sim_pdbs/ (7 PDB)    relaxed/ (T7c 弛豫产物)
test/module_results/
  report.md  t1_stage6.json  t2_quality_gate.json  t3_t4_mpnn.json
  t5_t6_scaling.json  t7_rosetta.json
```
