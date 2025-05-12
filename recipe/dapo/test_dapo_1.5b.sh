#!/usr/bin/env bash
set -xeuo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 # RZ: Override the max_position_embeddings for -Math-1.5B and -Math-7B models (4096 for the math models). This allows the model to accept context lengths larger than the model’s default maximum. But this seems does not work.

# module default
# module load cuda/12.6.1
# module load gcc/11.4.0
# conda init
# conda activate verl
wandb login 363018e9dc8339fae726d3b48a839f262c457194

project_name='DAPO'
exp_name='1.5B-vanilla-DAPO'

adv_estimator=rloo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=3072
max_num_batched_tokens=8192
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True # Whether we filter the prompts base on the pass rates.
filter_groups_metric=acc # The metric to filter the prompts.
max_num_gen_batches=50 # The maximum number of generations to generate. If we exceed this number, we will stop generating and raise error.
train_prompt_bsz=64
gen_prompt_bsz=320
train_prompt_mini_bsz=32
n_resp_per_prompt=20
n_resp_continue=0
n_resp_per_prompt_val=4
total_epochs=10
enable_curriculum=False
val_before_train=True
save_freq=-1
max_ckpt_to_keep=2

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# Paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B"}
use_chat_template=False
val_only=False

CKPT_PATH=${CKPT_PATH:-"/work/nvme/beok/rzhang15/checkpoints/DAPO/${exp_name}/$(date +%Y%m%d_%H%M%S)"}
mkdir -p ${CKPT_PATH}
TRAIN_FILE=${TRAIN_FILE:-"./data/DAPO-unique-Qwen-base/train.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- 
PYTHONUNBUFFERED=1 python3 -m recipe.dapo.src.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=[\"./data/DAPO-unique-Qwen-base/test.parquet\",\"./data/math500-Qwen-base/test.parquet\",\"./data/AIME-Qwen-base/train.parquet\"] \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.truncation='left' \
    data.use_chat_template=${use_chat_template} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.n_continue=${n_resp_continue} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=2 \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=disable \
    +trainer.val_only=${val_only} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.max_critic_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.save_metrics_local_dir=${WORKING_DIR}/metrics \
    trainer.save_metric_path=${exp_name}_$(date +%Y%m%d_%H%M%S) \
    trainer.default_local_dir=${CKPT_PATH} \
    curriculum.enable=${enable_curriculum} | tee ./logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log

# run
# ./recipe/dapo/test_dapo_1.5b.sh

# Run with nohup and redirect output to log file
# nohup ./recipe/dapo/test_dapo_1.5b.sh > ./logs/test_dapo_1.5b.log 2>&1 &
