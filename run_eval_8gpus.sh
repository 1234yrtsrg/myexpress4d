#!/bin/bash

# ================= Configuration =================
MODEL_PATH="./output_model/mdm_mydata/model000450000.pt"
DATASET="express4d"
DATA_MODE="arkit"
EVAL_MODE="wo_mm"
EVAL_MODEL="tex_mot_match"
RESULT_DIR="./eval_result"
NUM_GPUS=8
TOTAL_EVALS=20
REPS_PER_JOB=1
# =================================================

mkdir -p "$RESULT_DIR"
rm -f "${RESULT_DIR}"/eval_*_seed*.log

watch_job_log() {
    local log_path="$1"
    local job_label="$2"
    local startup_reported=0
    local sampling_reported=0

    while true; do
        if [ -f "$log_path" ]; then
            if grep -q "Traceback" "$log_path"; then
                echo "  [${job_label}] detected an error early. Check: ${log_path}"
                return
            fi
            if [ "$startup_reported" -eq 0 ] && grep -q "creating data loader..." "$log_path"; then
                echo "  [${job_label}] startup is normal; data loader stage reached."
                startup_reported=1
            fi
            if [ "$sampling_reported" -eq 0 ] && grep -q "Generated Dataset Loading Completed!!!" "$log_path"; then
                echo "  [${job_label}] sampling finished; metric evaluation is starting."
                sampling_reported=1
            fi
            if grep -q "========== Evaluating Matching Score ==========" "$log_path"; then
                echo "  [${job_label}] metric evaluation is running."
                return
            fi
            if grep -q "!!! DONE !!!" "$log_path"; then
                echo "  [${job_label}] evaluation finished."
                return
            fi
        fi
        sleep 15
    done
}

report_batch_status() {
    local logs=("$@")
    local completed=0
    local failed=0

    for log_path in "${logs[@]}"; do
        local log_name
        log_name=$(basename "$log_path")
        if grep -q "Traceback" "$log_path"; then
            echo "  [${log_name}] failed. See traceback in the log."
            failed=$((failed + 1))
        elif grep -q "!!! DONE !!!" "$log_path"; then
            echo "  [${log_name}] completed successfully."
            completed=$((completed + 1))
        else
            echo "  [${log_name}] finished without a final marker. Please inspect the log."
        fi
    done

    echo "Batch summary: ${completed} completed, ${failed} failed."
}

echo "Results will be saved to: $RESULT_DIR"
echo "Launching ${TOTAL_EVALS} evaluation jobs across ${NUM_GPUS} GPUs..."

running_jobs=0
batch_logs=()

for ((job_idx=0; job_idx<${TOTAL_EVALS}; job_idx++)); do
    gpu_id=$(( job_idx % NUM_GPUS ))
    seed=$(( (job_idx + 1) * 1000 ))
    log_name="eval_${EVAL_MODE}_seed${seed}.log"
    log_path="${RESULT_DIR}/${log_name}"
    job_label="job $((job_idx + 1))/${TOTAL_EVALS}, seed=${seed}, gpu=${gpu_id}"

    echo "Launching job $((job_idx + 1))/${TOTAL_EVALS} on GPU ${gpu_id} (seed=${seed})..."
    echo "  log file: ${log_path}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python -m eval.eval_humanml \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --data_mode "$DATA_MODE" \
        --device 0 \
        --cond_mode text \
        --eval_mode "$EVAL_MODE" \
        --eval_model_name "$EVAL_MODEL" \
        --eval_dataset_override express4d \
        --eval_rep_times "$REPS_PER_JOB" \
        --seed "$seed" > "$log_path" 2>&1 &

    watch_job_log "$log_path" "$job_label" &

    running_jobs=$((running_jobs + 1))
    batch_logs+=("$log_path")

    if [ "$running_jobs" -eq "$NUM_GPUS" ]; then
        echo "Waiting for the current batch of ${NUM_GPUS} jobs to finish..."
        wait
        report_batch_status "${batch_logs[@]}"
        running_jobs=0
        batch_logs=()
    fi
done

wait
if [ "${#batch_logs[@]}" -gt 0 ]; then
    report_batch_status "${batch_logs[@]}"
fi
echo "All evaluation jobs have completed."

echo "Aggregating results from all log files..."

python3 -c '
import glob
import math
import os
import re
import sys
import numpy as np

result_dir = sys.argv[1]
model_path = sys.argv[2]
log_files = sorted(glob.glob(os.path.join(result_dir, "eval_*_seed*.log")))

if not log_files:
    print("Error: no evaluation log files found.")
    sys.exit(1)

metrics = {
    "Matching Score": {"ground truth": [], "vald": []},
    "FID": {"ground truth": [], "vald": []},
    "Diversity": {"ground truth": [], "vald": []},
    "MultiModality": {"ground truth": [], "vald": []},
}
r_precision = {"ground truth": [], "vald": []}

for log_file in log_files:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_summary = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        summary_match = re.match(r"========== (.+) Summary ==========", line)
        if summary_match:
            current_summary = summary_match.group(1)
            continue

        if current_summary in metrics:
            metric_match = re.match(r"---> \[(ground truth|vald)\] Mean: ([0-9eE+\-.]+) CInterval: ([0-9eE+\-.]+)", line)
            if metric_match:
                label = metric_match.group(1)
                metrics[current_summary][label].append(float(metric_match.group(2)))
                continue

        if current_summary == "R_precision":
            rp_match = re.match(
                r"---> \[(ground truth|vald)\]\(top 1\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);"
                r"\(top 2\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);"
                r"\(top 3\) Mean: ([0-9eE+\-.]+) CInt: ([0-9eE+\-.]+);",
                line,
            )
            if rp_match:
                label = rp_match.group(1)
                r_precision[label].append([float(rp_match.group(2)), float(rp_match.group(4)), float(rp_match.group(6))])

def summarize(values):
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cint = 1.96 * std / math.sqrt(len(arr))
    return mean, cint

print(f"\n==================== Aggregated results for {os.path.basename(model_path)} from {len(log_files)} jobs ====================")

for metric_name in ("Matching Score", "R_precision", "FID", "Diversity", "MultiModality"):
    print(f"========== {metric_name} Summary ==========")
    if metric_name == "R_precision":
        for label in ("ground truth", "vald"):
            mean, cint = summarize(r_precision[label])
            if mean is not None:
                print(
                    f"---> [{label}](top 1) Mean: {mean[0]:.4f} CInt: {cint[0]:.4f};"
                    f"(top 2) Mean: {mean[1]:.4f} CInt: {cint[1]:.4f};"
                    f"(top 3) Mean: {mean[2]:.4f} CInt: {cint[2]:.4f};"
                )
    else:
        for label in ("ground truth", "vald"):
            mean, cint = summarize(metrics[metric_name][label])
            if mean is not None:
                if np.isscalar(mean):
                    print(f"---> [{label}] Mean: {mean:.4f} CInterval: {cint:.4f}")
                else:
                    print(f"---> [{label}] Mean: {mean[0]:.4f} CInterval: {cint[0]:.4f}")

print("=======================================================================================================================")
' "$RESULT_DIR" "$MODEL_PATH"
