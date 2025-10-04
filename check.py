from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
print("EOS token:", tok.eos_token)
print("EOS token id:", tok.eos_token_id)
