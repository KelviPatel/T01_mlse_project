from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_MODEL = None
_LLM_TOKENIZER = None

# You can change this to another instruct model if you want later
DEFAULT_LLM_NAME = "microsoft/phi-2"  # small-ish, general model


def load_llm(model_name: str = DEFAULT_LLM_NAME, device: Optional[str] = None):
    """
    Lazily loads a causal LM and tokenizer.
    You can swap model_name to another instruct-tuned model later.
    """
    global _LLM_MODEL, _LLM_TOKENIZER

    if _LLM_MODEL is not None and _LLM_TOKENIZER is not None:
        return _LLM_MODEL, _LLM_TOKENIZER

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    model.to(device)

    _LLM_MODEL = model
    _LLM_TOKENIZER = tokenizer

    return model, tokenizer


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """
    Simple helper to generate text from the LLM using a system + user prompt.

    For now this is plain concatenation; later we can adapt formatting
    if you switch to a chat-style model.
    """
    model, tokenizer = load_llm()
    device = next(model.parameters()).device

    # Simple prompt format for phi-2 or generic causal LMs
    full_prompt = f"{system_prompt.strip()}\n\nUser: {user_prompt.strip()}\nAssistant:"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt part, return only the completion after "Assistant:"
    if "Assistant:" in generated:
        generated = generated.split("Assistant:", 1)[1].strip()

    return generated
