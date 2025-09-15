#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR

source /projects/bffc/myang13/miniconda3/bin/activate verl

cd /u/myang13/eft-verl
EXPERIMENT_NAME=e3-rl-set-final-expanded
MODEL_PATH=/projects/bffc/myang13/e3-1.7B
PROJECT_NAME=e3-rl

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/projects/bffc/myang13/data/e3-rl-set-final-expanded.parquet \
    data.val_files=/projects/bffc/myang13/data/hmmt-aime-2025.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.35 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=17408 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=17408 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=17408 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=17408 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    custom_reward_function.path=verl/utils/reward_score/math_verify.py \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.default_local_dir=/projects/bdok/myang13/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.nnodes=4 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 "${@:1}" > /u/myang13/eft-verl/logs/$EXPERIMENT_NAME.log 2>&1


    # to enable DAPO Dyanmic Sampling...

    # algorithm.filter_groups.enable=True \
    # algorithm.filter_groups.metric=acc \
    # algorithm.filter_groups.max_num_gen_batches=10 \
    # data.gen_batch_size=256 \