#!/bin/bash
#SBATCH -J debug
#SBATCH -p defq
#SBATCH -A ece-research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H100:4
#SBATCH --mem=512G
#SBATCH --time=3-23:00:00
#SBATCH -o logs/%j.log

conda activate jspo
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

export CACHE=/your/path/to/cache/

export WANDB_API_KEY={your_wandb_key}
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1

unset ROCR_VISIBLE_DEVICES

MODEL_NAME=Qwen/Qwen3-4B-Base
MODEL_ID="${MODEL_NAME##*/}"

ROLLOUT_N=4

NUM_LORA_ADAPTERS=0
LORA_RANK=0
LOG_POLICY_GRAD=False
LEARNING_RATE=1e-6

LORA_ALPHA=$LORA_RANK
DATA_SEED=${2:-0}
ROLLOUT_SEED=${DATA_SEED}
MICRO_BATCH_SIZE=4
TEMPERATURE=1.0
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95
TEST_INTERVAL=10
SAVE_INTERVAL=20

ADVANTAGE_ESTIMATOR=${1:-rloo}
CORRECT_SAMPLE_LOG_PROB_COEF=0
INCORRECT_SAMPLE_LOG_PROB_COEF=0
JS_SMOOTH_COEF=0.99

export CUBLAS_WORKSPACE_CONFIG=:4096:8    
export NCCL_ALGO=Ring
export PYTHONHASHSEED=0

echo "job is starting on `hostname`"

CLIP_HIGH=0.28

PROJECT_NAME=debug-1020
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_math_ds${DATA_SEED}_rs${ROLLOUT_SEED}_lr${LEARNING_RATE}

echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=[$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet,$CACHE/verl-data/dapo17k/train.parquet] \
 data.val_files=[$CACHE/verl-data/aime24/test.parquet,$CACHE/verl-data/aime25/test.parquet] \
 data.train_batch_size=1024 \
 data.max_prompt_length=1024 \
 data.max_response_length=6144 \
 data.filter_overlong_prompts=True \
 data.chat_template_name=qwen-math \
 data.dataloader_num_workers=16 \
 data.truncation='error' \
 data.shuffle=True \
 data.seed=${DATA_SEED} \
 actor_rollout_ref.model.path=$CACHE/hf_models/$MODEL_NAME \
 actor_rollout_ref.model.use_shm=True \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
 actor_rollout_ref.actor.clip_ratio_low=0.2 \
 actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
 actor_rollout_ref.actor.loss_agg_mode=token-mean \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.kl_loss_coef=0.000 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
 actor_rollout_ref.rollout.top_k=-1 \
 actor_rollout_ref.rollout.top_p=1.00 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.rollout.n=${ROLLOUT_N} \
 actor_rollout_ref.rollout.seed=${ROLLOUT_SEED} \
 actor_rollout_ref.rollout.layered_summon=True \
 actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
 actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
 actor_rollout_ref.rollout.val_kwargs.n=16 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
 actor_rollout_ref.model.num_lora_adapters=${NUM_LORA_ADAPTERS} \
 actor_rollout_ref.model.lora_rank=${LORA_RANK} \
 actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
 actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.model.target_modules=all-linear \
 algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
 algorithm.js_smooth_coef=${JS_SMOOTH_COEF} \
 algorithm.use_kl_in_reward=False \
 algorithm.kl_ctrl.kl_coef=0.000 \
 algorithm.correct_sample_log_prob_coef=${CORRECT_SAMPLE_LOG_PROB_COEF} \
 algorithm.incorrect_sample_log_prob_coef=${INCORRECT_SAMPLE_LOG_PROB_COEF} \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 actor_rollout_ref.actor.log_policy_gradient=$LOG_POLICY_GRAD \
 trainer.save_freq=${SAVE_INTERVAL} \
 trainer.max_actor_ckpt_to_keep=10 \
 trainer.test_freq=${TEST_INTERVAL} \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=True \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.total_epochs=10 \
 ray_init.temp_dir=$CACHE/ray_tmp 