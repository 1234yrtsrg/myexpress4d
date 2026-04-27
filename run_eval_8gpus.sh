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

echo "Results will be saved to: $RESULT_DIR"
echo "Launching ${TOTAL_EVALS} evaluation jobs across ${NUM_GPUS} GPUs..."

running_jobs=0

for ((job_idx=0; job_idx<${TOTAL_EVALS}; job_idx++)); do
    gpu_id=$(( job_idx % NUM_GPUS ))
    seed=$(( (job_idx + 1) * 1000 ))
    log_name="eval_${EVAL_MODE}_seed${seed}.log"

    echo "Launching job $((job_idx + 1))/${TOTAL_EVALS} on GPU ${gpu_id} (seed=${seed})..."

    CUDA_VISIBLE_DEVICES=${gpu_id} python -m eval.eval_humanml \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --data_mode "$DATA_MODE" \
        --device 0 \
        --cond_mode text \
        --eval_mode "$EVAL_MODE" \
        --eval_model_name "$EVAL_MODEL" \
        --eval_rep_times "$REPS_PER_JOB" \
        --seed "$seed" > "${RESULT_DIR}/${log_name}" 2>&1 &

    running_jobs=$((running_jobs + 1))

    if [ "$running_jobs" -eq "$NUM_GPUS" ]; then
        wait
        running_jobs=0
    fi
done

wait
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

def parse_metric_mean(block_text, label):
    match = re.search(rf"\\[{re.escape(label)}\\] Mean: ([\\d\\.]+) CInterval: ([\\d\\.]+)", block_text)
    if match:
        return float(match.group(1))
    return None

for log_file in log_files:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for metric_name in metrics:
        block = re.search(rf"========== {re.escape(metric_name)} Summary ==========(.*?)(==========|$)", content, re.S)
        if not block:
            continue
        block_text = block.group(1)
        for label in ("ground truth", "vald"):
            value = parse_metric_mean(block_text, label)
            if value is not None:
                metrics[metric_name][label].append(value)

    rp_block = re.search(r"========== R_precision Summary ==========(.*?)(==========|$)", content, re.S)
    if rp_block:
        block_text = rp_block.group(1)
        for label in ("ground truth", "vald"):
            match = re.search(
                rf"\\[{re.escape(label)}\\]\\(top 1\\) Mean: ([\\d\\.]+) CInt: ([\\d\\.]+);"
                rf"\\(top 2\\) Mean: ([\\d\\.]+) CInt: ([\\d\\.]+);"
                rf"\\(top 3\\) Mean: ([\\d\\.]+) CInt: ([\\d\\.]+);",
                block_text
            )
            if match:
                r_precision[label].append([float(match.group(1)), float(match.group(3)), float(match.group(5))])

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
