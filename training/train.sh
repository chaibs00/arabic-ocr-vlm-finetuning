#!/bin/bash
# Step 6: Fine-tune Gemma-3 with LoRA using LlamaFactory.
# Run from the repo root:  bash training/train.sh
set -e

# 1. Install LlamaFactory at the tested commit (first run only)
if [ ! -d "LlamaFactory" ]; then
  git clone https://github.com/hiyouga/LlamaFactory.git
  cd LlamaFactory
  git checkout 762b480131908d37736ad9aa3f12e87f8f7e6313
  pip install -e . -r requirements/metrics.txt
  cd ..
fi

# 2. Load HF_TOKEN from .env and log in (Gemma is a gated model)
source .env
hf auth login --token "$HF_TOKEN"

# 3. Train
export DISABLE_VERSION_CHECK=1
llamafactory-cli train training/ocr_finetune_lora.yaml