#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量用 huggingface_hub 直传大目录到 HF：
  源: /home/azanette/JSPO/checkpoints/significance-test-1016
  目标: guanning-ai/significance-test-1016

特点
- 无需 git / git-lfs
- 以 create_commit 分批推送，断点友好
- 双阈值限流：每批最大文件数、最大字节数
- 可忽略某些目录/后缀
"""

import os
import sys
import math
import time
import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from huggingface_hub import HfApi, create_repo
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

# ========= 可配置 =========
LOCAL_DIR = Path("/home/azanette/JSPO/checkpoints/debug-1004")
REPO_ID = "guanning-ai/1024-1.5b-knk23-debug1004"
REPO_TYPE = "model"            # 如需 dataset 仓库，改成 "dataset"

BRANCH = "main"

# 批次阈值（满足其一即提交一次）
MAX_FILES_PER_COMMIT = 800           # 每批最多文件数
MAX_BYTES_PER_COMMIT = 1_200_000_000 # 每批最多总字节数（约 1.2 GB）

# 忽略规则（前缀=目录/文件名开头；后缀=扩展名）
IGNORE_DIR_PREFIXES = {".git", "__pycache__", ".ipynb_checkpoints"}
IGNORE_SUFFIXES = {".log", ".tmp", ".DS_Store"}

# 若文件很大，HF会自动用后端LFS存储；这里不需要本地 git-lfs。
# =========================

def should_ignore(path: Path, base: Path) -> bool:
    rel = path.relative_to(base)
    # 忽略规则：在任一级目录命中前缀则跳过
    for part in rel.parts[:-1]:
        if any(part.startswith(pre) for pre in IGNORE_DIR_PREFIXES):
            return True
    name = rel.name
    if any(name.endswith(suf) for suf in IGNORE_SUFFIXES):
        return True
    return False

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and not should_ignore(p, root):
            yield p

def gather_batches(files: List[Path]) -> List[List[Path]]:
    """按双阈值切批。"""
    batches: List[List[Path]] = []
    cur: List[Path] = []
    cur_bytes = 0
    for f in files:
        size = f.stat().st_size
        need_split = (
            len(cur) >= MAX_FILES_PER_COMMIT or
            (cur_bytes + size) > MAX_BYTES_PER_COMMIT
        )
        if need_split and cur:
            batches.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(f)
        cur_bytes += size
    if cur:
        batches.append(cur)
    return batches

def default_commit_message(batch_idx: int, total_batches: int, nfiles: int, nbytes: int) -> str:
    gb = nbytes / (1024**3)
    return f"upload batch {batch_idx}/{total_batches}: {nfiles} files (~{gb:.2f} GB)"

def ensure_repo(repo_id: str, repo_type: str, token: str):
    create_repo(repo_id=repo_id, token=token, repo_type=repo_type, exist_ok=True, private=False)

def remote_existing_paths(api: HfApi, repo_id: str, revision: str) -> Dict[str, int]:
    """
    获取远端已有文件列表（仅路径+大小），用于跳过已存在且大小一致的文件。
    注意：无法 100% 保证内容一致（同尺寸不同内容），
    但对 checkpoint 类文件通常足够实用。
    """
    paths = {}
    try:
        for info in api.list_repo_files_info(repo_id=repo_id, revision=revision):
            if info.type == "file" and info.path:
                paths[info.path] = info.size or 0
    except Exception:
        pass
    return paths

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("错误：未检测到环境变量 HF_TOKEN。请先 `export HF_TOKEN=你的HF令牌`", file=sys.stderr)
        sys.exit(1)

    if not LOCAL_DIR.exists():
        print(f"错误：目录不存在：{LOCAL_DIR}", file=sys.stderr)
        sys.exit(2)

    ensure_repo(REPO_ID, REPO_TYPE, token)
    api = HfApi(token=token)

    # 收集文件
    all_files = sorted(iter_files(LOCAL_DIR))
    print(f"[INFO] 本地文件总数：{len(all_files)}")

    # 远端已有文件（用于快速跳过相同大小的路径）
    print("[INFO] 正在读取远端已有文件信息（用于断点续传）...")
    remote_map = remote_existing_paths(api, REPO_ID, BRANCH)
    print(f"[INFO] 远端已存在文件：{len(remote_map)}")

    # 过滤掉“远端已存在且大小相同”的文件
    to_upload: List[Path] = []
    skipped = 0
    for f in all_files:
        rel = str(f.relative_to(LOCAL_DIR).as_posix())
        size = f.stat().st_size
        if remote_map.get(rel, None) == size:
            skipped += 1
            continue
        to_upload.append(f)
    print(f"[INFO] 需要上传：{len(to_upload)}，跳过相同大小文件：{skipped}")

    if not to_upload:
        print("[DONE] 没有需要上传的内容。")
        return

    batches = gather_batches(to_upload)
    total_batches = len(batches)
    print(f"[INFO] 计划分成 {total_batches} 个批次。")

    for i, batch in enumerate(batches, start=1):
        ops = []
        total_bytes = 0
        for f in batch:
            repo_path = str(f.relative_to(LOCAL_DIR).as_posix())
            ops.append(
                CommitOperationAdd(
                    path_in_repo=repo_path,
                    path_or_fileobj=str(f)  # 直接从本地路径读取
                )
            )
            total_bytes += f.stat().st_size

        msg = default_commit_message(i, total_batches, len(batch), total_bytes)
        print(f"[COMMIT] {msg}")

        # 提交（create_commit 会自动分块&重试，大文件在后端以 LFS 方式保存，无需本地 git-lfs）
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=BRANCH,
            operations=ops,
            commit_message=msg,
        )
        print(f"[OK] 已完成批次 {i}/{total_batches}：{len(batch)} 个文件")

    print("[DONE] 全部批次上传完成。")

if __name__ == "__main__":
    main()
