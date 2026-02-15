#!/bin/bash
# =============================================================================
# 使用预训练权重 weights.pth 在 mix 数据集上做分类微调（支持多卡）
# 使用前请：
#   1) 将 weights.pth 放到 PCAP_encoder/models/pretrained/ 下
#   2) 若尚未生成 parquet，先按下面 SPLIT_BY 在 PCAP_encoder 根目录运行一次：
#        packet（按包划分，有 data leakage）:
#          python Preprocess/mix_pcap_to_parquet.py --mix_dir /path/to/mix --out_dir 1.Datasets/Classification/mix --split_by packet
#        flow（按流划分，论文推荐）:
#          python Preprocess/mix_pcap_to_parquet.py --mix_dir /path/to/mix --out_dir 1.Datasets/Classification/mix_flow --split_by flow
#   3) 本脚本根据 SPLIT_BY 自动选择对应数据目录（mix 或 mix_flow）
# 执行方式：cd 到 PCAP_encoder 后运行
#   cd Experiments/5_classification_training && bash run_finetune_mix.sh
# =============================================================================

# -----------------------------------------------------------------------------
# 任务与日志
# -----------------------------------------------------------------------------
TASK="supervised"                    # 任务类型：supervised / self_supervision / inference
LOG_LEVEL="info"                     # 日志级别：debug / info / warning
OUTPUT_PATH="./results/"             # 结果与 checkpoint 输出目录（相对当前工作目录）

# -----------------------------------------------------------------------------
# 模型与 tokenizer
# -----------------------------------------------------------------------------
FINETUNED_PATH_MODEL="../../models/pretrained"   # 预训练权重目录，内含 weights.pth（相对本脚本所在目录）
# T5 模型与分词器：使用本地目录（因无法直连 HuggingFace）。须先下载 t5-base 到 models/t5-base，见下方说明
MODEL_NAME="../../models/t5-base"   # 本地 t5-base 目录（相对本脚本）
TOKENIZER_NAME="../../models/t5-base"
# 若尚未下载：在有网环境执行（或本机设 HF_ENDPOINT=https://hf-mirror.com 后执行）:
#   cd PCAP_encoder && python scripts/download_t5_base.py
# 再将生成的 models/t5-base 拷到本机对应位置

# -----------------------------------------------------------------------------
# GPU 配置（4 张 V100）
# -----------------------------------------------------------------------------
# 使用的 GPU 编号列表；accelerate 会为每个进程分配一张卡，一般填 0 到 N-1 即可
GPU=(0 1 2 3)
# 传给 Python 的 --gpu 参数，逗号分隔，如 "0,1,2,3"
GPU_STRING="$(IFS=, ; echo "${GPU[*]}")"
# 单机多卡时进程数 = GPU 数量；与 accelerate launch --num_processes 一致
export GPUS_PER_NODE=4

# -----------------------------------------------------------------------------
# 模型结构
# -----------------------------------------------------------------------------
BOTTLENECK="mean"                    # 编码器到分类头的压缩方式：none / first / mean / Luong

# -----------------------------------------------------------------------------
# 训练超参数
# -----------------------------------------------------------------------------
BATCH_SIZE=24                        # 每张 GPU 的 batch size；4 卡时有效 batch = 24*4 = 96
EPOCHS=2                            # 训练轮数
LR=0.001                             # 学习率（若解冻 encoder 可适当减小，如 0.0001）
MAX_QST_LENGTH=512                   # 输入（question+context）最大 token 数，超长截断
MAX_ANS_LENGTH=32                    # 答案/标签侧最大长度（分类任务中多为占位）
PERC=100                            # 使用训练/验证集的比例 1–100，100 表示用全量
SEED=43                              # 随机种子，保证划分与初始化可复现
LOSS="normal"                        # 损失权重：normal 等权 / weighted 按类别样本数加权
PKT_REPR_DIM=768                     # 包表示维度，与 T5-base 隐藏层一致

# -----------------------------------------------------------------------------
# 数据划分方式与路径（相对 PCAP_encoder 根目录）
# -----------------------------------------------------------------------------
# packet = 按包随机划分（Per-Packet Split，同一流可能同时出现在 train/test，存在 data leakage）
# flow   = 按流划分（Per-Flow Split，同一流只在一侧，论文推荐的严谨评估）
SPLIT_BY="flow"
if [ "$SPLIT_BY" = "flow" ]; then
    DATASET_ROOT="../../1.Datasets/Classification/mix_flow"
else
    DATASET_ROOT="../../1.Datasets/Classification/mix"
fi
TRAINING_DATA="${DATASET_ROOT}/train.parquet"
VAL_DATA="${DATASET_ROOT}/val.parquet"
TEST_DATA="${DATASET_ROOT}/test.parquet"

# -----------------------------------------------------------------------------
# 脚本与实验命名（IDENTIFIER 含 split 便于区分结果）
# -----------------------------------------------------------------------------
export SCRIPT=../../2.Training/classification/classification.py
IDENTIFIER="mix_${SPLIT_BY}_lr${LR}_seed${SEED}_batch${BATCH_SIZE}"
EXPERIMENT="mix_finetune"

# -----------------------------------------------------------------------------
# 传给 classification.py 的参数（与上面变量对应）
# -----------------------------------------------------------------------------
export SCRIPT_ARGS=" \
  --identifier $IDENTIFIER \
  --experiment $EXPERIMENT \
  --task $TASK \
  --clean_start \
  --tokenizer_name $TOKENIZER_NAME \
  --lr $LR \
  --loss $LOSS \
  --fix_encoder \
  --model_name $MODEL_NAME \
  --log_level $LOG_LEVEL \
  --output_path $OUTPUT_PATH \
  --training_data $TRAINING_DATA \
  --validation_data $VAL_DATA \
  --testing_data $TEST_DATA \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --seed $SEED \
  --bottleneck $BOTTLENECK \
  --max_qst_length $MAX_QST_LENGTH \
  --max_ans_length $MAX_ANS_LENGTH \
  --percentage $PERC \
  --gpu $GPU_STRING \
  --finetuned_path_model $FINETUNED_PATH_MODEL \
"
# 参数说明：
#   --identifier / --experiment  本次实验名，用于日志和保存路径
#   --clean_start                 若存在同 identifier 的旧结果则先清空再跑
#   --fix_encoder                 是否冻结 encoder（只训分类头），建议先 true 再尝试 false
#   --finetuned_path_model        加载预训练权重的目录（内含 weights.pth）

# -----------------------------------------------------------------------------
# 启动训练（4 进程 = 4 张 GPU，由 accelerate 自动做数据并行）
# -----------------------------------------------------------------------------
# 必须在 PCAP_encoder 根目录下执行，例如：
#   cd /path/to/Debunk_Traffic_Representation/code/PCAP_encoder
#   cd Experiments/5_classification_training && bash run_finetune_mix.sh
# 显式传参，避免 "The following values were not passed..." 的提示
accelerate launch \
  --num_processes=$GPUS_PER_NODE \
  --multi_gpu \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  $SCRIPT $SCRIPT_ARGS
