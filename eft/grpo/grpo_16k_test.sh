#!/bin/bash

source ~/.bashrc

source /work/10913/myang13/miniconda3/etc/profile.d/conda.sh
conda activate verl2

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR

EXPERIMENT_NAME=e3-1.7B-test-multinode
MODEL_PATH=/work/10913/myang13/models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
cd /home1/10913/myang13/verl


echo "--- DEBUGGING RAY JOB ENVIRONMENT ---" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
echo "Python executable: $(which python3)" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
python -c "import torch; print(f'Torch version: {torch.__version__}')" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
python -c "import triton; print(f'Triton version: {triton.__version__}')" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
nvcc -V >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
ulimit -a >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log
echo "-----------------------------------" >> /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME-debug.log


PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/work/10913/myang13/data/gsm8k/train.parquet \
 data.val_files=/work/10913/myang13/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=4 \
 trainer.save_freq=-1 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee /home1/10913/myang13/verl/logs/$EXPERIMENT_NAME.log

