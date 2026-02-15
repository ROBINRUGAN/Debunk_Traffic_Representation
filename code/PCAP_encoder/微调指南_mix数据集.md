# PCAP_encoder 项目结构 + 使用预训练权重在 mix 数据集上微调指南

## 一、项目（PCAP_encoder）代码结构

```
Debunk_Traffic_Representation/
├── code/
│   └── PCAP_encoder/                    # 即 pcapencoder 主代码
│       ├── 1.Datasets/                   # 数据集目录（需自行准备或从 HF 下载）
│       │   ├── Classification/          # 分类任务：train/val/test.parquet
│       │   ├── QA/                       # 问答预训练
│       │   └── Denoiser/                 # 去噪预训练
│       ├── 2.Training/                   # 训练入口
│       │   ├── classification/          # 分类微调
│       │   │   ├── classification.py   # 主入口
│       │   │   ├── flow_classification.py
│       │   │   └── inference.py
│       │   ├── QA/                       # QA 训练
│       │   └── Denoiser/                 # 去噪训练
│       ├── Core/                         # 核心实现
│       │   ├── classes/
│       │   │   ├── classification_model.py  # 分类模型，加载 weights.pth
│       │   │   ├── T5_model.py
│       │   │   ├── dataset_for_classification.py  # 读 parquet：context, class, question, type_q
│       │   │   ├── tokenizer.py
│       │   │   └── ...
│       │   └── functions/
│       │       ├── option_parser.py      # 命令行参数（含 --finetuned_path_model）
│       │       └── ...
│       ├── Experiments/                  # 现成实验脚本
│       │   └── 5_classification_training/
│       │       └── hpc_classification.sh  # 分类微调示例（用预训练）
│       ├── Preprocess/                   # 从 PCAP 生成数据集
│       └── environment.yml               # conda 环境 pcapencoder
└── process_finetune_data/               # 微调数据预处理
    └── Data Processing/
        └── PCAPEncoder/
            └── pcapencoder_process_pkt.ipynb  # 生成 Classification 所需 parquet
```

**要点：**

- **分类微调**：用 `2.Training/classification/classification.py`，通过 `--finetuned_path_model` 指向**包含 `weights.pth` 的目录**即可从预训练权重开始微调。
- **数据格式**：Classification 使用的 parquet 需包含列：`question`、`class`（类别索引）、`type_q`（类别名）、`context`（包十六进制，每 4 字符一空格一般为 `every4`）。

---

## 二、你已完成的准备

- 已下载预训练权重：`weights.pth`。
- 数据集在 **`llm-network/dataset/`** 下，有四种：`aes-128-gcm`、`aes-256-gcm`、`chacha20-poly1305`、**`mix`**。其中 **mix** 为按类别分文件夹的 pcap（如 `mix/coinbase.com/*.pcap`），需先转为 parquet 再微调。

---

## 三、接下来要做的事

### 1. 将 mix 的 pcap 转为 parquet（一次性）

你的 mix 数据在 **`/home/gxy/llm-network/dataset/mix`**，当前是「每类一个文件夹、里面若干 .pcap 文件」的形式。PCAP_encoder 需要的是 **train.parquet / val.parquet / test.parquet**（含列 `question`, `class`, `type_q`, `context`）。

在 **PCAP_encoder 根目录**下执行（会按 8:1:1 划分 train/val/test 并写入 `1.Datasets/Classification/mix/`）：

```bash
cd /home/gxy/llm-network/Debunk_Traffic_Representation/code/PCAP_encoder
python Preprocess/mix_pcap_to_parquet.py --mix_dir /home/gxy/llm-network/dataset/mix --out_dir 1.Datasets/Classification/mix
```

依赖：`scapy`、`pandas`、`pyarrow`（若缺可 `pip install scapy pandas pyarrow`）。  
生成文件：`1.Datasets/Classification/mix/train.parquet`、`val.parquet`、`test.parquet`、`mix.json`。

### 2. 放置预训练权重

在 PCAP_encoder 目录下建一个目录（例如 `models/pretrained`），**把 `weights.pth` 放进去**，例如：

```bash
cd /home/gxy/llm-network/Debunk_Traffic_Representation/code/PCAP_encoder
mkdir -p models/pretrained
# 把你下载的 weights.pth 复制到这里
cp /home/gxy/llm-network/Debunk_Traffic_Representation/weights.pth models/pretrained/
```

代码里加载方式是：`torch.load(f"{model_finetuned_path}/weights.pth")`，所以**必须是「目录 + weights.pth」**，不能只写文件路径。

### 3. 确认数据路径

完成步骤 1 后，数据应在：

- 训练：`1.Datasets/Classification/mix/train.parquet`
- 验证：`1.Datasets/Classification/mix/val.parquet`
- 测试：`1.Datasets/Classification/mix/test.parquet`

微调脚本 `run_finetune_mix.sh` 已默认指向上述路径（相对 `Experiments/5_classification_training` 的 `../../1.Datasets/Classification/mix`）。

### 4. 运行分类微调（单次运行示例）

在 **PCAP_encoder** 根目录下执行（或在 `Experiments/5_classification_training` 下用相对路径调整）：

```bash
cd /home/gxy/llm-network/Debunk_Traffic_Representation/code/PCAP_encoder

# 激活环境
conda activate pcapencoder

# 单次微调（按你的路径改 training/validation/testing_data 和 finetuned_path_model）
accelerate launch --num_processes=1 \
  2.Training/classification/classification.py \
  --task supervised \
  --identifier mix_finetune_lr0.001_seed43 \
  --experiment mix_finetune \
  --clean_start \
  --tokenizer_name T5-base \
  --model_name T5-base \
  --finetuned_path_model ./models/pretrained \
  --bottleneck mean \
  --lr 0.001 \
  --loss normal \
  --fix_encoder \
  --log_level info \
  --output_path ./results/ \
  --training_data ./1.Datasets/Classification/mix/train.parquet \
  --validation_data ./1.Datasets/Classification/mix/val.parquet \
  --testing_data ./1.Datasets/Classification/mix/test.parquet \
  --epochs 20 \
  --batch_size 24 \
  --seed 43 \
  --max_qst_length 512 \
  --max_ans_length 32 \
  --percentage 100 \
  --gpu 0
```

**参数说明：**

- `--finetuned_path_model ./models/pretrained`：指向放有 `weights.pth` 的目录。
- `--fix_encoder`：冻结 encoder，只训练分类头（更快、更稳）；若想去掉则不加该参数（会微调整个模型）。
- `--training_data / --validation_data / --testing_data`：改成你实际的 mix 的 parquet 路径。
- 若没有单独验证集，可把 `--validation_data` 去掉或设为空，代码会从训练集中划分 20% 做验证。

### 5. 使用自带脚本（推荐）

直接使用已写好的 **`Experiments/5_classification_training/run_finetune_mix.sh`**（已指向 `llm-network/dataset/mix` 转换后的输出路径）：

```bash
cd /home/gxy/llm-network/Debunk_Traffic_Representation/code/PCAP_encoder
conda activate pcapencoder
cd Experiments/5_classification_training
bash run_finetune_mix.sh
```

---

## 四、小结

| 步骤 | 内容 |
|------|------|
| 1 | 运行 `Preprocess/mix_pcap_to_parquet.py`，把 **`/home/gxy/llm-network/dataset/mix`** 的 pcap 转为 `1.Datasets/Classification/mix/` 下的 train/val/test.parquet |
| 2 | 把 `weights.pth` 放到 `code/PCAP_encoder/models/pretrained/weights.pth` |
| 3 | 在 PCAP_encoder 下执行 `cd Experiments/5_classification_training && bash run_finetune_mix.sh` 跑微调 |

按上述步骤即可用现有预训练权重在 **llm-network/dataset/mix** 上跑 pcapencoder 的分类微调。
