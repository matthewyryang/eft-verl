cd /work/10913/myang13/gh-packages
pip install triton-3.0.0-cp310-cp310-linux_aarch64.whl
pip install xformers-0.0.30+836cd905.d20250331-cp310-cp310-linux_aarch64.whl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl
pip install vllm-0.8.5.dev455+gd6484ef3c.d20250504.cu124-cp310-cp310-linux_aarch64.whl

conda install -c conda-forge libstdcxx-ng libgcc-ng

pip uninstall -y transformers
pip install "transformers>=4.41,<4.53"

cd ~/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
