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
      <img src="./static/images/teaser_figure.png" width="800" alt="Teaser Image">
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

We use VeRL framework for experiments. Please refer to the [VeRL documentation](https://verl.readthedocs.io/en/latest/index.html) and [Installation](https://verl.readthedocs.io/en/latest/start/install.html) for installation and usage.

We provide a quick start for the installation here.
```
conda create -n speed-rl python=3.10
conda activate speed-rl
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib

git clone https://github.com/Zanette-Labs/speed-rl.git
cd speed-rl
pip install -e .
```

## Data Preparation

We use NuminaMath, DAPO-17k, and DeepScalR for training and we use 1000 held-out data from DAPO-17k, MATH500, AMC23, AIME2024, AIME2025 for evaluation. For example, to preprocess the DeepScaleR dataser, please run:
```
python ./examples/data_preprocess/DeepScaleR.py
```
For other datasets, you can write your own script to preprocess the data following the format in `./examples/data_preprocess/DeepScaleR.py`. To get the training and testing data in our paper, please refer to the [data](https://huggingface.co/collections/rqzhang/speed-rl-684a72dfb24ea72540c32fa1).

Remember when you update the data, you need to update the reward functions in `_default_compute_score()` in `verl/utils/reward_score/__init__.py`. In this paper, we use Math-Verify to compute the reward for every training and testing example. When training a base model (instead of an instruct- model), you need to remove the chat template in the prompt and set `use_chat_template=False` in the training script.

## Training and Evaluation

SPEED can be combined with any rule-based RL algorithm. In this paper, we use RLOO and DAPO as the base RL algorithms. We use RLOO vs SPEED-RLOO as an example here.

To train the model via RLOO, please run:
```
./run_rloo_slurm.sh
```

To train the model via SPEED-RLOO, please run:
```
./run_fast_rloo_slurm.sh
```

## Key Hyper-parameters and Ways to Edit the Code
There are some hyper-parameters that you can edit in the training script.

- `curriculum.enable`: Whether to enable online curriculum learning method SPEED.
- `algorithm.filter_groups.enable`: This is the old hyperparameter for DAPO (not for our algorithm). If you want to run vanilla-DAPO, you need to set this to `True`.
- `data.train_batch_size`: The training batch size: the number of prompts used in one training step.
- `data.gen_batch_size`: The generation batch size: the number of prompts used in one generation step.
- `actor_rollout_ref.rollout.n`: The number of responses generated for each prompt in the **screening phase**.
- `actor_rollout_ref.rollout.n_continue`: The number of responses generated for each prompt in the **continuation phase**.

For other hyper-parameters, please refer to the [VeRL documentation](https://verl.readthedocs.io/en/latest/examples/config.html). Notice that, since the VeRL codebase is kept updated, you need to check the correct version of the codebase for the exact meaning of the hyper-parameters since we did not use the latest version of the VeRL.

The main engineering edits for the SPEED method lies in

- `recipe/dapo/src/fast_dapo_ray_trainer.py`: The trainer class for SPEED. To edit the training loop, you need to edit the `fit()` function.
- `verl/trainer/ppo/data_controller.py`: The data controller class for SPEED. This controls the management of the data batches used for inference and for training.

## Acknowledgement

This work has greatly benefited from the use of Delta’s advanced computing and data resource
supported by the National Science Foundation (OAC 2005572) and the State of Illinois. Overall, this
project used ACCESS grants CIS250428 for its compute resources.

The authors gratefully acknowledges <a href="https://tajwarfahim.github.io/">Fahim Tajwar</a>, <a href="https://sheikhshafayat.github.io/">Sheikh Shafayat</a> and all the other members in Zenatte's Lab for their helpful suggestions and valuable feedback.

## Bibtex 

```
@misc{zhang2025speedrlfastertrainingreasoning,
      title={SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning}, 
      author={Ruiqi Zhang and Daman Arora and Song Mei and Andrea Zanette},
      year={2025},
      eprint={2506.09016},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.09016}, 
} 
```


