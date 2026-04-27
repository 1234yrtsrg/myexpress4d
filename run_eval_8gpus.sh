# ================= 配置区 =================
MODEL_PATH="./output_model/mdm_mydata/model000450000.pt"
DATASET="express4d"
DATA_MODE="arkit"
EVAL_MODE="wo_mm"
EVAL_MODEL="tex_mot_match"
RESULT_DIR="./eval_result"
# ==========================================

# 1. 准备工作：创建结果文件夹
mkdir -p $RESULT_DIR
echo "📁 结果将保存在: $RESULT_DIR"
echo "🚀 开始 8 卡满载并行评估..."

# 2. 启动 8 个并行进程
for i in {0..7}; do
    SEED=$(( (i+1) * 1000 ))
    LOG_NAME="eval_${EVAL_MODE}_seed${SEED}.log"
    
    echo "启动 GPU $i (Seed: $SEED, Log: $LOG_NAME)..."
    
    CUDA_VISIBLE_DEVICES=$i python -m eval.eval_humanml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --data_mode $DATA_MODE \
        --device 0 \
        --cond_mode text \
        --eval_mode $EVAL_MODE \
        --eval_model_name $EVAL_MODEL \
        --seed $SEED > "${RESULT_DIR}/${LOG_NAME}" 2>&1 &
done

# 3. 等待所有后台进程结束
wait
echo "✅ 所有 GPU 评估任务已完成！"

# 4. 自动统计结果并高度还原官方格式
echo "📊 正在聚合所有日志文件的结果..."

python3 -c '
import sys
import glob
import re
import os
import numpy as np

result_dir = sys.argv[1]
model_path = sys.argv[2]
log_files = glob.glob(os.path.join(result_dir, "eval_*_seed*.log"))

if not log_files:
    print("❌ 错误：未找到日志文件！")
    sys.exit(1)

metrics = {
    "Matching Score": {"gt": [], "vald": []},
    "FID": {"gt": [], "vald": []},
    "Diversity": {"gt": [], "vald": []},
    "MultiModality": {"gt": [], "vald": []}
}
r_prec = {"gt": [], "vald": []}

# 逐个文件提取数据
for f in log_files:
    with open(f, "r") as file:
        content = file.read()
        
        # 提取基础指标 (包含 Mean 和 CInterval)
        def extract_basic(metric_name):
            block = re.search(rf"========== {metric_name} Summary ==========(.*?)(==========|$)", content, re.S)
            if block:
                text = block.group(1)
                gt = re.search(r"\[ground truth\] Mean: ([\d\.]+) CInterval: ([\d\.]+)", text)
                if gt: metrics[metric_name]["gt"].append([float(gt.group(1)), float(gt.group(2))])
                
                vald = re.search(r"\[vald\] Mean: ([\d\.]+) CInterval: ([\d\.]+)", text)
                if vald: metrics[metric_name]["vald"].append([float(vald.group(1)), float(vald.group(2))])
                
        extract_basic("Matching Score")
        extract_basic("FID")
        extract_basic("Diversity")
        extract_basic("MultiModality")
        
        # 提取 R_precision (包含 top 1, 2, 3 的 Mean 和 CInt)
        rp_block = re.search(r"========== R_precision Summary ==========(.*?)(==========|$)", content, re.S)
        if rp_block:
            text = rp_block.group(1)
            gt = re.search(r"\[ground truth\]\(top 1\) Mean: ([\d\.]+) CInt: ([\d\.]+);\(top 2\) Mean: ([\d\.]+) CInt: ([\d\.]+);\(top 3\) Mean: ([\d\.]+) CInt: ([\d\.]+)", text)
            if gt: r_prec["gt"].append([float(x) for x in gt.groups()])
            
            vald = re.search(r"\[vald\]\(top 1\) Mean: ([\d\.]+) CInt: ([\d\.]+);\(top 2\) Mean: ([\d\.]+) CInt: ([\d\.]+);\(top 3\) Mean: ([\d\.]+) CInt: ([\d\.]+)", text)
            if vald: r_prec["vald"].append([float(x) for x in vald.groups()])

print(f"\n==================== 模型 {os.path.basename(model_path)} 全局平均结果 ({len(log_files)}个进程联合求均) ====================")

# 格式化输出基础指标
def print_basic(metric_name):
    gt = np.mean(metrics[metric_name]["gt"], axis=0) if metrics[metric_name]["gt"] else None
    vald = np.mean(metrics[metric_name]["vald"], axis=0) if metrics[metric_name]["vald"] else None
    
    if gt is not None or vald is not None:
        print(f"========== {metric_name} Summary ==========")
        if gt is not None: 
            print(f"---> [ground truth] Mean: {gt[0]:.4f} CInterval: {gt[1]:.4f}")
        if vald is not None: 
            print(f"---> [vald] Mean: {vald[0]:.4f} CInterval: {vald[1]:.4f}")

print_basic("Matching Score")

# 格式化输出 R_precision
gt_rp = np.mean(r_prec["gt"], axis=0) if r_prec["gt"] else None
vald_rp = np.mean(r_prec["vald"], axis=0) if r_prec["vald"] else None
if gt_rp is not None or vald_rp is not None:
    print("========== R_precision Summary ==========")
    if gt_rp is not None: 
        print(f"---> [ground truth](top 1) Mean: {gt_rp[0]:.4f} CInt: {gt_rp[1]:.4f};(top 2) Mean: {gt_rp[2]:.4f} CInt: {gt_rp[3]:.4f};(top 3) Mean: {gt_rp[4]:.4f} CInt: {gt_rp[5]:.4f};")
    if vald_rp is not None: 
        print(f"---> [vald](top 1) Mean: {vald_rp[0]:.4f} CInt: {vald_rp[1]:.4f};(top 2) Mean: {vald_rp[2]:.4f} CInt: {vald_rp[3]:.4f};(top 3) Mean: {vald_rp[4]:.4f} CInt: {vald_rp[5]:.4f};")

print_basic("FID")
print_basic("Diversity")
print_basic("MultiModality")
print("=======================================================================================================================")
' "$RESULT_DIR" "$MODEL_PATH"