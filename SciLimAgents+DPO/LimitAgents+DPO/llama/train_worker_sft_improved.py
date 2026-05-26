#!/usr/bin/env python3
"""Worker SFT LoRA training for Llama/Mistral/Qwen chat models."""

import sys

from train_sft_lora import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--agent-name", "worker"]
    main() 

    