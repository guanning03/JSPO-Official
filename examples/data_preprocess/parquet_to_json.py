import pandas as pd
from pathlib import Path

# List all your parquet files here
data_dir_lst = [
    # "./data/AIME-Qwen-base/train.parquet",
    # "./data/AIME2025-Qwen-base/train.parquet",
    # "./data/DAPO-unique-Qwen-base/train.parquet",
    # "./data/DAPO-unique-Qwen-base/test.parquet",
    # "./data/gsm8k-Qwen-base/train.parquet",
    # "./data/gsm8k-Qwen-base/test.parquet",
    # "./data/math500-Qwen-base/train.parquet",
    # "./data/math500-Qwen-base/test.parquet",
    # "./data/AMC-Qwen-base/train.parquet",
    # "./data/math-base/train.parquet",
    # "./data/math-base/test.parquet",
    # "./data/AIME2025-instruct/train.parquet",
    # "./data/AIME2024-instruct/train.parquet",
    # "./data/DAPO-17k-instruct/train.parquet",
    # "./data/DAPO-17k-instruct/test.parquet",
    # "./data/gsm8k-instruct/train.parquet",
    # "./data/gsm8k-instruct/test.parquet",
    # "./data/Math500-instruct/train.parquet",
    # "./data/Math500-instruct/test.parquet",
    # "./data/Math-instruct/train.parquet",
    # "./data/Math-instruct/test.parquet",
    # "./data/AMC23-instruct/train.parquet"
    # "./data/NuminaMath-base/train.parquet"
    # "./data/NuminaMath-processed-instruct/train.parquet",
    # "./data/merged4-instruct/train.parquet"
    # "./data/NuminaMath-allsources-base/train.parquet"
    # "./data/Numinamath-filtered-instruct/train.parquet"
    # "./data/DeepScaleR-Qwen-base/train.parquet"
    # "./data/DeepScaleR-instruct/train.parquet"
]

for parquet_path in data_dir_lst:
    parquet_path = Path(parquet_path)
    
    # 1. read parquet
    df = pd.read_parquet(parquet_path)
    
    # 2. build output path by swapping .parquet → .jsonl
    jsonl_path = parquet_path.with_suffix(".jsonl")
    
    # 3. write JSON Lines
    df.to_json(jsonl_path, orient="records", lines=True)
    
    print(f"Converted {parquet_path.name} → {jsonl_path.name}")
