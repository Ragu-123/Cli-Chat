# model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_IDENTIFIER = "HuggingFaceTB/SmolLM2-360M-Instruct"

def initialize_model():
    """
    Loads the model and tokenizer with CPU-friendly settings.
    """
    print(f"Initializing model: {MODEL_IDENTIFIER}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDENTIFIER,
        device_map="cpu",
        dtype=torch.float32,        # fixed
        low_cpu_mem_usage=True
    )

    print("Model and tokenizer initialized successfully.")
    return model, tokenizer
