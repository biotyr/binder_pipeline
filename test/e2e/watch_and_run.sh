#!/bin/bash
# 候卡 watcher (参数化): 检测到空闲 >=22500MiB 的显卡后, 连续跟踪 5 分钟,
# 期间空间持续充足才提交对应 e2e 任务 (run_e2e.py 完整 6 阶段)。
# 用法: watch_and_run.sh <task> <hotspot>
#   例:  watch_and_run.sh 3RRQ_single with
# 注: 2026-09-14 起 GPU1 (PCI 82:00.0) 驱动级故障 ("Unknown Error"),
#     全局 nvidia-smi 会整体报错 — 必须逐卡 -i 探测并跳过故障卡。
set -u
TASK=${1:?usage: watch_and_run.sh <task> <hotspot>}
HS=${2:?usage: watch_and_run.sh <task> <hotspot>}
PY=/home/taoyr/anaconda3/envs/esmbind/bin/python
DIR=/home/taoyr/projects/pipeline
# 空闲阈值可被环境变量覆盖: 4ZQK 多链 (233aa+binder≤200, ~433 token ≈ 任务一
# 287 token 的 1.5×) 激活显存增长, 实测任务一峰值 allocated 17.94GiB →
# nvidia-smi ~20.5-21.5GiB, 22.3G 空闲对多链太贴边, 链脚本对 4ZQK 用 23500。
THRESH=${THRESH:-22500}   # MiB 空闲阈值
TRACK_SECONDS=300     # 确认窗口: 5 分钟
POLL=30               # 跟踪期检查间隔 (秒)
MAX_TRIES=720         # 扫描期上限: 720 x 60s = 12h

log() { echo "$(date '+%F %T') [$TASK/$HS] $*"; }

free_of() {           # 指定卡空闲 MiB; 探测失败或非数字 → 空串
    local v
    v=$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    case $v in ''|*[!0-9]*) echo "";; *) echo "$v";; esac
}

find_gpu() {          # 输出第一个达标卡 index; 无则 return 1
    local g v
    for g in 0 1 2 3; do
        v=$(free_of "$g")
        [ -n "$v" ] && [ "$v" -ge "$THRESH" ] && { echo "$g"; return 0; }
    done
    return 1
}

uuid_of() {           # 指定卡的 GPU UUID。CUDA_VISIBLE_DEVICES 必须用 UUID:
                      # GPU1 死掉后 CUDA runtime 只枚举 3 个设备 (ordinal 0/1/2),
                      # nvidia-smi 物理 index 3 无对应 ordinal, 用数字会
                      # "No CUDA GPUs are available" (2026-09-14 三连崩根因)。
    nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '
}

for i in $(seq 1 $MAX_TRIES); do
    GPU=$(find_gpu) || GPU=""
    if [ -n "$GPU" ]; then
        log "检测到 GPU ${GPU} 空闲达标 (>=${THRESH}MiB), 开始 ${TRACK_SECONDS}s 跟踪"
        ok=1
        for ((s=POLL; s<=TRACK_SECONDS; s+=POLL)); do
            sleep $POLL
            FREE=$(free_of "$GPU")
            if [ -z "$FREE" ] || [ "$FREE" -lt "$THRESH" ]; then
                log "GPU ${GPU} 跟踪 ${s}s 时回落 (${FREE:-探测失败}MiB), 放弃该卡, 重新扫描"
                ok=0
                break
            fi
        done
        if [ $ok -eq 1 ]; then
            FREE=$(free_of "$GPU")
            UUID=$(uuid_of "$GPU")
            if [ -z "$UUID" ]; then
                log "GPU ${GPU} UUID 获取失败, 放弃本轮, 重新扫描"
            else
                log "GPU ${GPU} 连续 ${TRACK_SECONDS}s 保持充足 (当前 ${FREE}MiB), 提交 e2e 任务 (CUDA_VISIBLE_DEVICES=$UUID)"
                cd "$DIR"
                # E2E_EXTRA_ARGS: 可选附加参数 (如 --epitope-weight 0.2), 默认空
                CUDA_VISIBLE_DEVICES=$UUID HF_HUB_OFFLINE=1 $PY test/e2e/run_e2e.py \
                    --task "$TASK" --hotspot "$HS" ${E2E_EXTRA_ARGS:-}
                rc=$?
                log "run_e2e 结束 rc=$rc"
                exit $rc
            fi
        fi
    fi
    if [ $((i % 30)) -eq 1 ]; then
        log "扫描 $i/$MAX_TRIES: 暂无达标卡 (free MiB: GPU0=$(free_of 0) GPU1=[故障] GPU2=$(free_of 2) GPU3=$(free_of 3))"
    fi
    sleep 60
done
log "12h 内未等到持续空闲的卡, 放弃"
exit 42
