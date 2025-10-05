from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_IDENTIFIER = "HuggingFaceTB/SmolLM2-360M-Instruct"

def initialize_model():
    """
    Load tokenizer and model onto CPU, create a text-generation pipeline, and return (model, tokenizer, gen_pipeline).
    """
    print(f"Initializing model: {MODEL_IDENTIFIER}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDENTIFIER,
        device_map="cpu",
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Model, tokenizer and pipeline initialized successfully.")
    return model, tokenizer, gen
