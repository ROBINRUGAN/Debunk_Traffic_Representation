#!/usr/bin/env python3
"""
将 HuggingFace 上的 t5-base 下载到本地，供无法直连 HuggingFace 时使用。
在有网络的机器上运行（或本机设 HF_ENDPOINT 镜像后运行）:
  pip install huggingface_hub
  python scripts/download_t5_base.py
下载完成后，在 run_finetune_mix.sh 中设置:
  MODEL_NAME="../../models/t5-base"
  TOKENIZER_NAME="../../models/t5-base"
"""
import os

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return

    # 保存到 PCAP_encoder/models/t5-base（相对本脚本，脚本在 scripts/ 下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(script_dir, "..", "models", "t5-base")
    os.makedirs(local_dir, exist_ok=True)
    print(f"下载 google-t5/t5-base 到: {os.path.abspath(local_dir)}")
    snapshot_download(repo_id="google-t5/t5-base", local_dir=local_dir)
    print("完成。在 run_finetune_mix.sh 中设置 MODEL_NAME 与 TOKENIZER_NAME 为该路径（相对 Experiments/5_classification_training 即 ../../models/t5-base）")

if __name__ == "__main__":
    main()
