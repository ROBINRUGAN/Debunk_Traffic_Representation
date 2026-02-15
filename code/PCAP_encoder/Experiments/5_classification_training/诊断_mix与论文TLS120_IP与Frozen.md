# Mix 数据集与论文 TLS-120 的对应关系 + Loss 不降诊断

## 1. 数据集对应关系

| 维度 | 论文 (TLS-120) | 你的任务 (mix) |
|------|----------------|----------------|
| 数据来源 | CSTN-TLS1.3，120 个网站 | mix，41 个网站 |
| 任务类型 | 包级分类，Per-flow split | 包级分类，Per-flow split（8:1:1） |
| **IP 地址** | 论文 Table 7 做了「保留 / 移除 IP」消融 | **你的 mix 在预处理里已抹掉 IP** |

## 2. 你的 mix 数据对 IP/端口的处理（关键）

在 `Preprocess/mix_pcap_to_parquet.py` 的 `clean_packet()` 中：

```python
# 当前实现：IP 和端口被统一抹掉
packet[scapy.IP].src = "0.0.0.0"
packet[scapy.IP].dst = "0.0.0.0"
# TCP/UDP 端口 → 0
packet[scapy.TCP].sport = 0
packet[scapy.TCP].dport = 0
```

因此，送入模型的 `context` 是**已脱敏的包**（无真实 IP、无真实端口），等价于论文中的 **w/o IP addr.** 设定。

## 3. 论文中的关键数字（与你直接相关）

- **Table 3（Per-flow split, Frozen encoder）**
  - Pcap-Encoder on TLS-120：**71.0%** AC，**63.7** macro F1（在**保留 IP** 的设定下）。

- **Table 7（Pcap-Encoder 消融，Frozen）**
  - **base**（保留 IP）：TLS-120 macro F1 = **63.7**
  - **w/o IP addr.**：TLS-120 macro F1 = **13.0**（接近随机猜）
  - **w/o header**：TLS-120 掉到 1.5

- **Table 4（Frozen vs Unfrozen）**
  - TLS-120：Frozen 71.0% AC / 63.7 F1 → Unfrozen **77.3%** AC / **69.2** F1

结论：在 Frozen 前提下，**一旦去掉 IP，Pcap-Encoder 在 TLS-120 上表现会崩到 ~13% F1**；你当前是 Frozen + 无 IP，和这一设定一致。

## 4. 对你「Loss 不降」的诊断

- 你当前配置：**fix_encoder=True**（只训分类头，Encoder 冻结）+ **mix 数据（无真实 IP、端口置 0）**。
- 这等价于论文中的：
  - **Frozen Encoder**
  - **w/o IP addr.**

在这种组合下，论文 Table 7 已经表明：**Frozen + 无 IP → TLS-120 只有 ~13% F1**。  
因此你看到 **Loss 停在 ~3.68、准确率接近随机**，与论文结论一致，并不是实现 bug，而是**设定与数据（无 IP）叠加后的预期现象**。

原因可以概括为：

1. **预训练更依赖包头/IP**：Pcap-Encoder 的 Q&A 预训练包含“目标 IP”等包头语义，Encoder 学到的表示里 IP 占比很大。
2. **Frozen 时表示固定**：冻结后，分类头只能基于当前表示；若表示里最有区分度的就是 IP，而你的数据里 IP 已被抹成 0.0.0.0，则分类头几乎得不到有效信号。
3. **特征与标签脱节**：类别与“真实 IP”的对应在你的数据里不存在（或变成常数），Frozen 表示无法区分 41 类，Loss 难以下降。

## 5. 建议（按优先级）

### 方案 A：解冻 Encoder（Unfreeze）——与论文 Table 4 一致

- 在 `run_finetune_mix.sh` 中去掉 `--fix_encoder`（或设为 false），让 Encoder 参与微调。
- 论文：TLS-120 上 Unfrozen 比 Frozen 提升明显（71.0 → 77.3% AC）；在**无 IP** 的情况下，解冻后模型可以重新从包头其他字段（以及时序等）学出可区分特征，而不是依赖“固定 IP 表示”。
- 注意：解冻后建议适当减小学习率（如 1e-4），并可能需要更多 epoch；你脚本里已有注释提示。

### 方案 B：保留 IP（若合规且可接受）

- 若你的场景允许使用真实 IP，可修改 `mix_pcap_to_parquet.py`：**不要**把 `IP.src/dst` 置为 0.0.0.0，保留原始值（或只做部分脱敏），这样数据设定就接近论文 **base**（有 IP），Frozen 才有机会达到论文中 ~71% 的水平。
- 端口是否保留视需求而定；论文消融主要针对 IP。

### 方案 C：继续 Frozen + 无 IP（仅作对照）

- 若目的只是复现论文「w/o IP + Frozen → 性能崩溃」的结论，当前配置已经能体现这一点（Loss 不降、准确率接近随机）。

## 6. 小结

- **你的 mix 数据 = 论文的「w/o IP addr.» 设定**（预处理已抹掉 IP 和端口）。
- **Frozen + w/o IP** 在论文里就对应 TLS-120 上 **~13% F1**，与你现在「Loss 不降、像瞎猜」一致。
- 若要在**不引入真实 IP** 的前提下提升效果，**解冻 Encoder（Unfreeze）** 是论文和实验都支持的首选方案；若可保留 IP，则 Frozen 也有机会达到论文中的 base 性能。
