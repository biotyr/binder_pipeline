#!/bin/bash
# 剩余 3 个 e2e 任务串行链:
#   每任务 = watcher 候卡 (5分钟稳定性跟踪) → run_e2e.py (完整 6 阶段 pipeline)
#   → 校验 pipeline 内 Stage 5 是否产出 dG; 仅缺失时才触发 rescore_stage56.py
#   (补救后备, 对应任务一的链名 bug 场景)。
# 进度写入 test/e2e/chain.log。
set -u
DIR=/home/taoyr/projects/pipeline
PY=/home/taoyr/anaconda3/envs/esmbind/bin/python
LOG=$DIR/test/e2e/chain.log

log() { echo "$(date '+%F %T') [chain] $*" | tee -a "$LOG"; }

dg_count() {   # pipeline 原生 Stage 5 产出的 dG 数 (selection.parquet 非空 rosetta_dG)
    "$PY" - "$1" <<'EOF'
import sys
import pandas as pd
try:
    df = pd.read_parquet(f"{sys.argv[1]}/full_pipeline/selection.parquet")
    print(int(df["rosetta_dG"].notna().sum()))
except Exception:
    print(0)
EOF
}

: > "$LOG"
log "启动: 3RRQ_single/with → 4ZQK_multi/no → 4ZQK_multi/with"

OK=0
FAIL=0
# 每任务: task hotspot 空闲阈值MiB — 3RRQ 单链 22500 (任务一同条件, 实证可跑);
# 4ZQK 多链 23500 (token 1.5×, 激活显存增长, 22.3G 太贴边)
for spec in "3RRQ_single with 22500" "4ZQK_multi no 23500" "4ZQK_multi with 23500"; do
    set -- $spec
    TASK=$1; HS=$2; THRESH=$3
    log "===== 任务开始: $TASK / $HS (阈值 ${THRESH}MiB) ====="
    THRESH=$THRESH bash "$DIR/test/e2e/watch_and_run.sh" "$TASK" "$HS" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 42 ]; then
        log "watcher 12h 超时, 终止整条链"
        exit 42
    fi
    if [ "$rc" -ne 0 ]; then
        log "run_e2e 失败 rc=$rc (见上方日志), 跳过本任务, 继续下一任务"
        FAIL=$((FAIL+1))
        continue
    fi

    TDIR=$DIR/test/results/$TASK/${HS}_hotspot
    N_DG=$(dg_count "$TDIR")
    if [ "$N_DG" -gt 0 ]; then
        log "pipeline 内 Stage 5 已产出 $N_DG 个 dG, 无需重评分 — 任务 $TASK/$HS 完成"
        OK=$((OK+1))
    else
        log "pipeline 内 Stage 5 未产出 dG, 启动补救重评分 (CPU, ~25min)"
        "$PY" "$DIR/test/e2e/rescore_stage56.py" --task "$TASK" --hotspot "$HS" 2>&1 | tee -a "$LOG"
        rc2=${PIPESTATUS[0]}
        if [ "$rc2" -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); fi
        log "rescore rc=$rc2 — 任务 $TASK/$HS 结束"
    fi
done
if [ "$OK" -eq 3 ]; then
    log "全部 3 个任务完成"
else
    log "链结束: 成功 $OK / 失败 $FAIL / 共 3 个任务 (失败任务见上方日志)"
fi
