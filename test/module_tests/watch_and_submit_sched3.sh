#!/bin/bash
# watcher + submit: 臂 ii 三阶段调度测试 (设计用于 setsid nohup 脱离 session 运行)
# 流程: 扫描空闲 GPU(<2000MiB) → 5x60s 稳定性跟踪 → 确认后 UUID 提交测试。
# 预算: 最多 2000 轮 (~24h), 成功提交并跑完即退出; 活动日志本文件 stdout。
THRESHOLD=2000; STABLE_CHECKS=5; MAX_ITER=2000
PY=/home/taoyr/anaconda3/envs/esmbind/bin/python
SCRIPT=/home/taoyr/projects/pipeline/test/module_tests/diag_inter_schedule3_ii.py
LOG=/home/taoyr/projects/pipeline/test/module_tests/diag_inter_schedule3_ii.log

for ((scan=1; scan<=MAX_ITER; scan++)); do
  for g in 0 1 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$used" ] && [ "$used" -lt $THRESHOLD ] 2>/dev/null; then
      echo "[$(date '+%m-%d %H:%M')] GPU$g 候选 (${used}MiB), 开始 ${STABLE_CHECKS}min 稳定性跟踪"
      stable=1
      for i in $(seq 1 $STABLE_CHECKS); do
        sleep 60
        used2=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        echo "[$(date '+%m-%d %H:%M')] 稳定性 $i/${STABLE_CHECKS}: ${used2:-ERR}MiB"
        if [ -z "$used2" ] || [ "$used2" -ge $THRESHOLD ] 2>/dev/null; then
          echo "[$(date '+%m-%d %H:%M')] GPU$g 稳定性失败, 回到扫描"
          stable=0; break
        fi
      done
      if [ "$stable" -eq 1 ]; then
        uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
        echo "[$(date '+%m-%d %H:%M')] GPU$g 稳定通过, 提交测试 (UUID=$uuid)"
        CUDA_VISIBLE_DEVICES="$uuid" HF_HUB_OFFLINE=1 "$PY" "$SCRIPT" > "$LOG" 2>&1
        echo "[$(date '+%m-%d %H:%M')] 测试结束 exit=$?"
        exit 0
      fi
    fi
  done
  sleep 60
done
echo "[$(date '+%m-%d %H:%M')] 24h 预算耗尽, 退出"
exit 1
