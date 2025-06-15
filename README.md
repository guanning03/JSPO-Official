<h1 style="text-align: center;">SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning</h1>

<div align="center">
This is the github repo for the paper "<strong>SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning</strong>" by Ruiqi Zhang, Daman Arora, Song Mei and Andrea Zanette. 👇
<br>
<a href="https://zanette-labs.github.io/speed-rl/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
<a href="https://arxiv.org/pdf/2506.09016">
    <img src="https://img.shields.io/badge/Paper-%23FF2442?style=for-the-badge&logo=bytedance&logoColor=white"></a>
<a href="https://github.com/Zanette-Labs/speed-rl">
    <img src="https://img.shields.io/badge/Tweet-%230084FF?style=for-the-badge&logo=bytedance&logoColor=white"></a>
<a href="https://x.com/ruiqizhang0614/status/1933527717036834843?s=12">
    <img src="https://img.shields.io/badge/Code-07C160?style=for-the-badge&logo=bytedance&logoColor=white"></a>
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

We use NuminaMath, DAPO-17k, and DeepScalR for training and we use 1000 held-out data from DAPO-17k, MATH500, AMC23, AIME2024, AIME2025 for evaluation. For example, to preprocess the data, please run:

```


```

## Training and Evaluation



## Edit the Code

## Acknowledgement
This work has greatly benefited from the use of Delta’s advanced computing and data resource
supported by the National Science Foundation (OAC 2005572) and the State of Illinois. Overall, this
project used ACCESS grants CIS250428 for its compute resources.

We thank Fahim Tajwar, Zhaoyi Zhou, Sheikh Shafayat, Guanning Zeng, Zitong Yang and Tianyu Guo for their helpful discussions and feedback. 

## Citation 
To cite this work, please use the following BibTeX entry:
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


