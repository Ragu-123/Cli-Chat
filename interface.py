import threading
import re
import torch
from transformers import TextIteratorStreamer
from model_loader import initialize_model
from chat_memory import ConversationBuffer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and clearly. "
    "your answer should be concise and factual."
    "Use recent chat history"
)

def build_prompt(history, user_message):
    """
    Construct the full prompt from the system prompt, recent history, and the current user message.
    """
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for role, text in history:
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)

def clean_reply(raw_reply: str) -> str:
    """
    Trim generation after user/system markers and remove optional leading 'Assistant:' label, returning a clean assistant response.
    """
    if not raw_reply:
        return ""
    stop_markers = [r"<\|im_start\|>user", r"<\|im_start\|>system"]
    cut = re.split("|".join(stop_markers), raw_reply, maxsplit=1)[0]
    return re.sub(r"^\s*(Assistant:)?\s*", "", cut).strip()

def stream_reply(model, tokenizer, prompt, max_new_tokens=256):
    """
    Stream generated tokens using TextIteratorStreamer and model.generate.
    Yields chunks (strings) as they become available.
    """
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_id = getattr(tokenizer, "eos_token_id", None)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=lambda: model.generate(**inputs, **kwargs), daemon=True)
    thread.start()
    for chunk in streamer:
        yield chunk
    thread.join()

def generate_reply(gen_pipeline, prompt, max_new_tokens=256):
    """
    Non-streaming generation using the Hugging Face pipeline.
    Returns the generated text (string).
    """
    outputs = gen_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    if isinstance(outputs, list) and outputs:
        return outputs[0].get("generated_text", outputs[0].get("text", ""))
    return ""

def begin_chat():
    """
    Start a CLI chat loop. Uses streaming generation via model.generate + TextIteratorStreamer.
    Falls back to pipeline non-streaming generation if streaming fails.
    """
    model, tokenizer, gen = initialize_model()
    memory = ConversationBuffer(window_size=3)

    print("\nChat session started. Type '/exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "/exit":
            print("Exiting. Goodbye!")
            break
        if not user_input:
            continue

        prompt = build_prompt(memory.get_history(), user_input)

        print("Bot: ", end="", flush=True)
        raw_chunks = []
        try:
            for chunk in stream_reply(model, tokenizer, prompt):
                print(chunk, end="", flush=True)
                raw_chunks.append(chunk)
            print()
            reply_text = "".join(raw_chunks)
            if not reply_text.strip():
                reply_text = generate_reply(gen, prompt)
                print(reply_text)
        except Exception:
            reply_text = generate_reply(gen, prompt)
            print(reply_text)

        reply = clean_reply(reply_text)
        memory.add_message("user", user_input)
        memory.add_message("assistant", reply)

if __name__ == "__main__":
    begin_chat()
