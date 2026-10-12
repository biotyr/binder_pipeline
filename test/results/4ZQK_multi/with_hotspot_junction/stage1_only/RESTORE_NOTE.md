# stage1_only 恢复记录 (2026-09-16)

- seed1 / w=0.05 / 150 步的 stage1 最优设计 (iPTM 0.8245, 过质量门) 曾被
  steps=300 实验 (15:39) 覆写 stage1_results.parquet 与原始 PDB。
- 幸存: relax 副本 (15:00, score_stage1_only.py 的产物) 与
  stage1_rosetta_scores.json (dG=-47.67 等全套 Rosetta 指标)。
- 本目录已重建: stage1_results.parquet (序列取自 relaxed PDB 链C, 94 aa;
  ptm/final_loss 无法恢复记 None), REMARK 已注入 relaxed PDB。
- 原始(未 relax)结构不可恢复; 300 步失败设计存档于 steps300_attempt/。
