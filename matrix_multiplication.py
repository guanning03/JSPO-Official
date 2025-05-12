#!/usr/bin/env python
import torch
import torch.multiprocessing as mp
import os

def worker(gpu_id):
    # bind this process to a single GPU
    torch.cuda.set_device(gpu_id)

    # allocate two large random matrices on this GPU
    size = 10_000
    x = torch.randn(size, size, device=f"cuda:{gpu_id}")
    y = torch.randn(size, size, device=f"cuda:{gpu_id}")

    # infinite loop to keep GPU busy
    iteration = 0
    while True:
        z = torch.matmul(x, y)
        # rotate buffers to avoid out-of-memory
        x, y = z, x
        iteration += 1
        if iteration % 100000 == 0:
            # optional heartbeat log
            print(f"[GPU {gpu_id}] completed {iteration} multiplies", flush=True)

if __name__ == "__main__":
    # detect all available GPUs (should be 4 under Slurm --gres=gpu:4)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No GPUs found. Make sure --gres=gpu:<N> is set.")

    # spawn one process per GPU
    mp.spawn(worker, nprocs=ngpus)
