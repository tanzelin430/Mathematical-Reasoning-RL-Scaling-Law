import gzip
import base64
import json
from datasets import load_dataset

# 1. 加载 train 切分
ds = load_dataset(
    "LLM360/guru-RL-92k-extra-info-compressed",
    split="train",
    streaming=False
)

def pad_b64(s: str) -> str:
    # 补全 Base64 的 = 填充
    s = s.strip()
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s

def decompress_str(b64: str) -> str:
    # Base64 解码 + gzip 解压
    raw = base64.b64decode(pad_b64(b64))
    return gzip.decompress(raw).decode("utf-8")

def unpack_fields(example):
    # —— 解压 extra_info —— #
    parts = example["extra_info"]
    # 按数字顺序拼回完整 Base64 字符串
    full_b64 = "".join(parts[f"part_{i}"] for i in range(1, len(parts) + 1))
    # 一次性解压出完整的 JSON 文本
    json_text = decompress_str(full_b64)
    # 转成 Python 对象
    example["extra_info_decoded"] = json.loads(json_text)

    # —— 解压 reward_model.ground_truth —— #
    gt_b64 = example["reward_model"]["ground_truth"]["compressed"]
    gt_json = json.loads(decompress_str(gt_b64))
    example["ground_truth_decoded"] = gt_json

    return example

# 4. 批量解压
ds2 = ds.map(unpack_fields)

# 5. 验证效果
print("Extra info (decoded):")
print(ds2[0]["extra_info_decoded"])
print("\nGround truth (decoded):")
print(ds2[0]["ground_truth_decoded"])