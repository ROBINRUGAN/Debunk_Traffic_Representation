#!/usr/bin/env python3
"""
将 llm-network/dataset/mix 目录下的 pcap（按类别分文件夹）转为 PCAP_encoder 分类微调所需的 parquet。
保留 IP 与端口等包头信息，以便 Frozen Encoder 能利用预训练学到的表示（与论文 TLS-120 base 设定一致）。

划分方式（--split_by）：
  packet: 按包随机划分（Per-Packet Split），同一流的包可能同时出现在 train/test，存在 data leakage。
  flow:   按流划分（Per-Flow Split），同一流的所有包只出现在 train/val/test 之一，论文推荐的严谨评估方式。

用法:
  python mix_pcap_to_parquet.py --mix_dir /path/to/mix --out_dir ./1.Datasets/Classification/mix --split_by packet
  python mix_pcap_to_parquet.py --mix_dir /path/to/mix --out_dir ./1.Datasets/Classification/mix_flow --split_by flow
"""
import os
import sys
import json
import argparse
import binascii

try:
    import scapy.all as scapy
except Exception as e:
    print("请安装 scapy: pip install scapy", file=sys.stderr)
    raise

import pandas as pd
from tqdm import tqdm


def clean_packet(packet):
    """只去掉以太网头，保留 IP、端口等包头信息，供 Pcap-Encoder 分类使用。"""
    if packet.haslayer(scapy.Ether):
        packet = packet[scapy.Ether].payload
    return packet


def get_flow_key(packet):
    """
    从 scapy 包计算流的唯一标识（5-tuple，双向等价）。
    用于 Per-Flow Split：同一流的包必须落在同一划分（train/val/test）中。
    无 IP 或 无 TCP/UDP 的包返回 None，将按单包成流处理。
    """
    if packet.haslayer(scapy.IP):
        ip_layer = packet[scapy.IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = int(ip_layer.proto)
        if packet.haslayer(scapy.TCP):
            src_port = int(packet[scapy.TCP].sport)
            dst_port = int(packet[scapy.TCP].dport)
        elif packet.haslayer(scapy.UDP):
            src_port = int(packet[scapy.UDP].sport)
            dst_port = int(packet[scapy.UDP].dport)
        else:
            return None
        ip_pair = tuple(sorted([str(src_ip), str(dst_ip)]))
        port_pair = tuple(sorted([src_port, dst_port]))
        return (ip_pair, port_pair, proto)
    if packet.haslayer("IPv6"):
        ip_layer = packet["IPv6"]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = int(ip_layer.nh)
        if packet.haslayer(scapy.TCP):
            src_port = int(packet[scapy.TCP].sport)
            dst_port = int(packet[scapy.TCP].dport)
        elif packet.haslayer(scapy.UDP):
            src_port = int(packet[scapy.UDP].sport)
            dst_port = int(packet[scapy.UDP].dport)
        else:
            return None
        ip_pair = tuple(sorted([str(src_ip), str(dst_ip)]))
        port_pair = tuple(sorted([src_port, dst_port]))
        return (ip_pair, port_pair, proto)
    return None


def group_string_by_n(pkt_bytes, n=4):
    s = binascii.hexlify(bytes(pkt_bytes)).decode()
    return " ".join(s[i : i + n] for i in range(0, len(s), n))


def main():
    parser = argparse.ArgumentParser(description="mix pcap -> PCAP_encoder parquet")
    parser.add_argument(
        "--mix_dir",
        type=str,
        default="/home/gxy/llm-network/dataset/mix",
        help="mix 数据集根目录，下为按类别名的子文件夹，每类文件夹内为 .pcap 文件",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./1.Datasets/Classification/mix",
        help="输出目录，将生成 train.parquet, val.parquet, test.parquet, mix.json",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例（其余一半 val 一半 test）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="随机划分的种子",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        choices=["packet", "flow"],
        default="packet",
        help="划分方式: packet=按包随机划分(存在 data leakage); flow=按流划分(同一流只在一侧，论文推荐)",
    )
    args = parser.parse_args()

    mix_dir = os.path.abspath(args.mix_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(mix_dir):
        print(f"错误: 目录不存在 {mix_dir}", file=sys.stderr)
        sys.exit(1)

    # 类别 = 子文件夹名（排除隐藏文件等）
    class_names = sorted(
        [
            d
            for d in os.listdir(mix_dir)
            if os.path.isdir(os.path.join(mix_dir, d)) and not d.startswith(".")
        ]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"类别数: {len(class_names)}, 类别: {class_names[:5]}...")

    QUESTION = "What is the representation of this packet?"
    rows = []
    split_by_flow = args.split_by == "flow"

    for type_q in tqdm(class_names, desc="类别"):
        class_dir = os.path.join(mix_dir, type_q)
        cls = class_to_idx[type_q]
        pcap_files = [
            f
            for f in os.listdir(class_dir)
            if f.endswith(".pcap") and not f.startswith(".")
        ]
        for fn in tqdm(pcap_files, desc=f"  {type_q[:20]}", leave=False):
            path = os.path.join(class_dir, fn)
            try:
                with scapy.PcapReader(path) as reader:
                    for pkt in reader:
                        try:
                            pkt = clean_packet(pkt)
                            context = group_string_by_n(pkt)
                            if split_by_flow:
                                fk = get_flow_key(pkt)
                                if fk is None:
                                    fk = ("_no_flow_", len(rows))
                                rows.append([QUESTION, cls, type_q, context, fk])
                            else:
                                rows.append([QUESTION, cls, type_q, context])
                        except Exception:
                            continue
            except Exception as e:
                tqdm.write(f"跳过 {path}: {e}")

    if split_by_flow:
        df = pd.DataFrame(rows, columns=["question", "class", "type_q", "context", "flow_id"])
    else:
        df = pd.DataFrame(rows, columns=["question", "class", "type_q", "context"])
    print(f"总包数: {len(df)}")

    # 8:1:1 划分
    if split_by_flow:
        # Per-Flow Split: 先按流分组，对流做划分，再按流归属把包归入 train/val/test
        rng = __import__("random").Random(args.seed)
        flow_ids = df["flow_id"].unique().tolist()
        rng.shuffle(flow_ids)
        nf = len(flow_ids)
        t = int(nf * args.train_ratio)
        v = (nf - t) // 2
        te = nf - t - v
        train_flows = set(flow_ids[:t])
        val_flows = set(flow_ids[t : t + v])
        test_flows = set(flow_ids[t + v :])
        train_df = df[df["flow_id"].isin(train_flows)].drop(columns=["flow_id"]).reset_index(drop=True)
        val_df = df[df["flow_id"].isin(val_flows)].drop(columns=["flow_id"]).reset_index(drop=True)
        test_df = df[df["flow_id"].isin(test_flows)].drop(columns=["flow_id"]).reset_index(drop=True)
        print(f"流数: {nf}, train流: {len(train_flows)}, val流: {len(val_flows)}, test流: {len(test_flows)}")
    else:
        # Per-Packet Split: 按包随机打乱后按行划分
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n = len(df)
        t = int(n * args.train_ratio)
        v = (n - t) // 2
        te = n - t - v
        train_df = df.iloc[:t]
        val_df = df.iloc[t : t + v]
        test_df = df.iloc[t + v :]

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    with open(os.path.join(out_dir, "mix.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"已写入: {out_dir}")
    print(f"  train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    main()
