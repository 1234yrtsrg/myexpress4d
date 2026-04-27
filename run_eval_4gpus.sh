#!/bin/bash

# 定义模型路径和其他参数
MODEL_PATH="./output_model/mdm_mydata/model000450000.pt"
DATASET="express4d"
DATA_MODE="arkit"
EVAL_MODE="wo_mm"
EVAL_MODEL="tex_mot_match"

echo "开始 4 卡并行评估..."

# 启动 GPU 0 (Seed 1000)
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_humanml \
    --model_path $MODEL_PATH --dataset $DATASET --data_mode $DATA_MODE \
    --device 0 --cond_mode text --eval_mode $EVAL_MODE \
    --eval_model_name $EVAL_MODEL --seed 1000 &

# 启动 GPU 1 (Seed 2000)
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_humanml \
    --model_path $MODEL_PATH --dataset $DATASET --data_mode $DATA_MODE \
    --device 0 --cond_mode text --eval_mode $EVAL_MODE \
    --eval_model_name $EVAL_MODEL --seed 2000 &

# 启动 GPU 2 (Seed 3000)
CUDA_VISIBLE_DEVICES=2 python -m eval.eval_humanml \
    --model_path $MODEL_PATH --dataset $DATASET --data_mode $DATA_MODE \
    --device 0 --cond_mode text --eval_mode $EVAL_MODE \
    --eval_model_name $EVAL_MODEL --seed 3000 &

# 启动 GPU 3 (Seed 4000)
CUDA_VISIBLE_DEVICES=3 python -m eval.eval_humanml \
    --model_path $MODEL_PATH --dataset $DATASET --data_mode $DATA_MODE \
    --device 0 --cond_mode text --eval_mode $EVAL_MODE \
    --eval_model_name $EVAL_MODEL --seed 4000 &

# 等待所有后台进程结束
wait

echo "所有 GPU 评估任务已完成！请检查生成的 4 个日志文件。"
