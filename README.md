<h1 align="center">James-Stein Policy Optimization</h1>

<div align="center">
This is the github repo for the paper "<strong>Variance-Reduced Reinforcement Learning for Large Reasoning Models via James-Stein Baselines</strong>" by <a href="https://github.com/guanning03">Guanning Zeng</a>, <a href="https://zhaoyizhou1123.github.io/">Zhaoyi Zhou</a>, <a href="https://daman1209arora.github.io/">Daman Aurora</a> and <a href="https://azanette.com/">Andrea Zanette</a>.
<br>
<br>
<a href="https://zanette-labs.github.io/speed-rl/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge"></a>
<a href="https://arxiv.org/pdf/2506.09016">
    <img src="https://img.shields.io/badge/Paper-%23FF2442?style=for-the-badge"></a>
<a href="https://github.com/Zanette-Labs/speed-rl">
    <img src="https://img.shields.io/badge/Code-%230084FF?style=for-the-badge"></a>
<a href="https://x.com/ruiqizhang0614/status/1933527717036834843?s=12">
    <img src="https://img.shields.io/badge/Tweet-07C160?style=for-the-badge"></a>
</div>

<table>
  <tr>
    <td align="center">
      <img src="./site/static/images/teaser_figure.png" width="800" alt="Teaser Image">
    </td>
  </tr>
  <tr>
    <td align="center">SPEED expends some compute (left figure, red region) to identify and exclude low-signal (low-SNR) prompts from the training batch, ensuring the majority of compute is effectively utilized on informative prompts. This yields an average 4× speedup across various benchmarks and training configurations (right figure; see paper for details).</td>
  </tr>
</table>

SPEED-RL is an online curriculum learning framework for accelerating rule-based RL training of reasoning models.

- **Algorithm Design**: We design an online curriculum learning framework that adaptively selects questions that are moderately difficult for training based on the model's performance.
It is widely applicable and can be combined with any rule-based RL algorithm.

- **Experiments**: We evaluate the wall-clock time to reach a certain accuracy on some reasoning benchmarks for the base RL algorithm (RLOO and DAPO) and its SPEED variant. SPEED can achieve on average 2x to 6x speedups over different base RL algorithms, training configurations, and evaluation benchmarks.

- **Theoretical Insights**: We provide a theoretical analysis of why the moderate difficulty questions are more informative for RL training by relatign them with the signal-to-noise ratio of the gradient estimate.

## Getting Started

Create a conda environment and install dependencies

```bash
conda create -n jspo python=3.10
conda activate jspo
USE_MEGATRON=0 bash misc/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd peft
pip install -v -e .
cd ..
pip install numpy==1.26 math_verify==0.8.0
```

Set up a global cache with at least 50GB ssd available

```bash
export CACHE=/path/to/your/cache
```

- `$CACHE`: Global Settings of the cache file, must be set to the absolute path without `~` or soft links
- `$CACHE/hf_models/{hf_id}/{hf_name}`: Default Path of model
- `$CACHE/verl-data/{dataset_name}/train.parquet(test.parquet)`: Default Path of Data

Download models and datasets, for example:

```bash
python misc/downlaod_model.py
python misc/download_knk_data.py --dataset="self-label-zanette-lab/knight-knave-3" --save_name="train"
python misc/download_knk_data.py --dataset="self-label-zanette-lab/knight-knave-3-OOD-test100" --save_name="test"
python misc/download_math_data.py --dataset="guanning-ai/dapo17k" --no_test
python misc/download_math_data.py --dataset="guanning/aime25" --no_train
```

Change `WANDB_TOKEN` and slurm setups in `run_knk.sh` as your own, and launch the experiments:

```bash
bash run_knk.sh jspo
bash run_knk.sh rloo            # baseline / comparison
```

## Acknowledgement

The authors gratefully acknowledges <a href="https://tajwarfahim.github.io/">Fahim Tajwar</a>, <a href="https://sheikhshafayat.github.io/">Sheikh Shafayat</a> and all the other members in Zenatte's Lab for their helpful suggestions and valuable feedback.

## Bibtex 

```
To Be Filled
```


