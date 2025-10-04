# interface.py
import threading, re
from transformers import TextIteratorStreamer
from model_loader import initialize_model
from chat_memory import ConversationBuffer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and clearly. "
    "your answer should be concise and factual."
    "Use recent chat history"
)

def build_prompt(history, user_message):
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
    for role, text in history:
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)

def clean_reply(raw_reply: str) -> str:
    if not raw_reply:
        return ""
    stop_markers = [r"<\|im_start\|>user", r"<\|im_start\|>system"]
    cut = re.split("|".join(stop_markers), raw_reply, maxsplit=1)[0]
    return re.sub(r"^\s*(Assistant:)?\s*", "", cut).strip()

def stream_reply(model, tokenizer, prompt, max_new_tokens=256):
    """
    Stream generated tokens deterministically (CPU-safe).
    """
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # CPU-friendly, deterministic
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,                   # deterministic, avoids ignored flags
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=lambda: model.generate(**inputs, **kwargs), daemon=True)
    thread.start()
    for chunk in streamer:
        yield chunk
    thread.join()

def begin_chat():
    model, tokenizer = initialize_model()
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
        for chunk in stream_reply(model, tokenizer, prompt):
            print(chunk, end="", flush=True)
            raw_chunks.append(chunk)
        print()

        reply = clean_reply("".join(raw_chunks))
        memory.add_message("user", user_input)
        memory.add_message("assistant", reply)

if __name__ == "__main__":
    begin_chat()
