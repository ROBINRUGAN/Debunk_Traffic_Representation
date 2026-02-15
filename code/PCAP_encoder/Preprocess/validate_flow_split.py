#!/usr/bin/env python3
"""
检验按流划分（Per-Flow Split）是否正确：
1. 用与 mix_pcap_to_parquet.py 相同的逻辑从 pcap 建 flow_id 并做 8:1:1 划分；
2. 检查 train/val/test 的流集合两两不交且并集为全部流；
3. 若提供 --out_dir，则与已生成的 parquet 行数对比，确保一致。

用法（在 PCAP_encoder 根目录或 Preprocess 下）:
  python Preprocess/validate_flow_split.py --mix_dir /path/to/mix --seed 43 --train_ratio 0.8
  python Preprocess/validate_flow_split.py --mix_dir /path/to/mix --out_dir 1.Datasets/Classification/mix_flow --seed 43 --train_ratio 0.8
"""
import os
import sys
import argparse

try:
    import scapy.all as scapy
except Exception:
    print("请安装 scapy: pip install scapy", file=sys.stderr)
    sys.exit(1)

import pandas as pd
from tqdm import tqdm

# 与 mix_pcap_to_parquet 保持一致的逻辑
def clean_packet(packet):
    if packet.haslayer(scapy.Ether):
        packet = packet[scapy.Ether].payload
    return packet

def get_flow_key(packet):
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


def build_flow_df(mix_dir, seed, train_ratio):
    """与 mix_pcap_to_parquet 相同的遍历与 flow_id 构建，返回 (df, train_flows, val_flows, test_flows)。"""
    mix_dir = os.path.abspath(mix_dir)
    class_names = sorted(
        [
            d
            for d in os.listdir(mix_dir)
            if os.path.isdir(os.path.join(mix_dir, d)) and not d.startswith(".")
        ]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    rows = []
    for type_q in tqdm(class_names, desc="类别", unit="类"):
        class_dir = os.path.join(mix_dir, type_q)
        cls = class_to_idx[type_q]
        pcap_files = [
            f for f in os.listdir(class_dir)
            if f.endswith(".pcap") and not f.startswith(".")
        ]
        for fn in tqdm(pcap_files, desc=f"  {type_q[:16]}", leave=False, unit="pcap"):
            path = os.path.join(class_dir, fn)
            try:
                with scapy.PcapReader(path) as reader:
                    for pkt in reader:
                        try:
                            pkt = clean_packet(pkt)
                            fk = get_flow_key(pkt)
                            if fk is None:
                                fk = ("_no_flow_", len(rows))
                            rows.append([cls, type_q, fk])
                        except Exception:
                            continue
            except Exception:
                continue
    sys.stdout.flush()
    df = pd.DataFrame(rows, columns=["class", "type_q", "flow_id"])
    # 与 mix_pcap_to_parquet 完全一致的划分
    rng = __import__("random").Random(seed)
    flow_ids = df["flow_id"].unique().tolist()
    rng.shuffle(flow_ids)
    nf = len(flow_ids)
    t = int(nf * train_ratio)
    v = (nf - t) // 2
    train_flows = set(flow_ids[:t])
    val_flows = set(flow_ids[t : t + v])
    test_flows = set(flow_ids[t + v :])
    return df, train_flows, val_flows, test_flows


def main():
    parser = argparse.ArgumentParser(description="Validate per-flow split")
    parser.add_argument("--mix_dir", type=str, default="/home/gxy/llm-network/dataset/mix")
    parser.add_argument("--out_dir", type=str, default="", help="若提供，与 train/val/test.parquet 行数对比")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("按流划分验证 (validate_flow_split)", flush=True)
    print("=" * 60, flush=True)
    print(f"mix_dir: {os.path.abspath(args.mix_dir)}", flush=True)
    print(f"seed={args.seed}, train_ratio={args.train_ratio}", flush=True)
    if args.out_dir:
        print(f"out_dir: {os.path.abspath(args.out_dir)} (将对比 parquet 行数)", flush=True)
    print("", flush=True)
    print("Step 1/2: 从 pcap 构建 flow_id 并做 8:1:1 划分...", flush=True)
    df, train_flows, val_flows, test_flows = build_flow_df(args.mix_dir, args.seed, args.train_ratio)
    n = len(df)
    nf = len(train_flows) + len(val_flows) + len(test_flows)
    print(f"  总包数: {n}, 流数: {nf}", flush=True)
    print("", flush=True)
    print("Step 2/2: 检验流划分与 parquet 一致性...", flush=True)

    # 1) 流集合两两不交
    train_val = train_flows & val_flows
    train_test = train_flows & test_flows
    val_test = val_flows & test_flows
    ok_disjoint = len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0
    print(f"[1] 流集合两两不交: {'通过' if ok_disjoint else '失败'}", flush=True)
    if not ok_disjoint:
        if train_val:
            print(f"    train ∩ val 非空: {len(train_val)} 个流", flush=True)
        if train_test:
            print(f"    train ∩ test 非空: {len(train_test)} 个流", flush=True)
        if val_test:
            print(f"    val ∩ test 非空: {len(val_test)} 个流", flush=True)

    # 2) 并集为全部流
    all_flows = train_flows | val_flows | test_flows
    ok_cover = len(all_flows) == nf and nf == len(df["flow_id"].unique())
    print(f"[2] 流并集 = 全部流 (共 {nf} 流): {'通过' if ok_cover else '失败'}", flush=True)

    # 3) 按流划分后的包数
    train_count = df["flow_id"].isin(train_flows).sum()
    val_count = df["flow_id"].isin(val_flows).sum()
    test_count = df["flow_id"].isin(test_flows).sum()
    print(f"[3] 按流划分包数: train={train_count}, val={val_count}, test={test_count}, 合计={train_count + val_count + test_count} (总包数 {n})", flush=True)
    ok_sum = (train_count + val_count + test_count) == n
    print(f"    包数合计与总包数一致: {'通过' if ok_sum else '失败'}", flush=True)

    ok_match = True
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
        train_path = os.path.join(out_dir, "train.parquet")
        val_path = os.path.join(out_dir, "val.parquet")
        test_path = os.path.join(out_dir, "test.parquet")
        if os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path):
            print(f"[4] 读取 parquet...", flush=True)
            tr = len(pd.read_parquet(train_path))
            va = len(pd.read_parquet(val_path))
            te = len(pd.read_parquet(test_path))
            print(f"    已生成 parquet 行数: train={tr}, val={va}, test={te}", flush=True)
            ok_match = (tr == train_count and va == val_count and te == test_count)
            print(f"    与本次划分一致: {'通过' if ok_match else '失败'}", flush=True)
            if not ok_match:
                print(f"    预期: train={train_count}, val={val_count}, test={test_count}", flush=True)
        else:
            print(f"[4] 未找到 {out_dir} 下 train/val/test.parquet，跳过行数对比", flush=True)

    all_ok = ok_disjoint and ok_cover and ok_sum and ok_match
    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"总体: {'全部通过' if all_ok else '存在失败项'}", flush=True)
    print("=" * 60, flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
