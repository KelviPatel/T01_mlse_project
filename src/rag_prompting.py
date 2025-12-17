from typing import List

from src.rag_index import retrieve_lore
from src.text_llm import generate_text

def format_lore_context(lore_results: List[dict]) -> str:
    """
    Combines retrieved lore chunks into a readable context block.
    """
    parts = []
    for i, item in enumerate(lore_results, start=1):
        text = item.get("text", "").strip()
        if not text:
            continue
        parts.append(f"[LORE {i}]\n{text}")
    return "\n\n".join(parts)

def rag_enrich_image_prompt(user_prompt: str, top_k: int = 3) -> str:
    """
    Uses RAG to enrich a user prompt with relevant lore,
    then asks the LLM to create a single, vivid image prompt.

    Returns: enriched prompt string.
    """
    # 1) Retrieve relevant lore
    lore_results = retrieve_lore(user_prompt, top_k=top_k)
    lore_context = format_lore_context(lore_results)

    # 2) Build system + user prompts for LLM
    system_prompt = (
        "You are an assistant that creates concise but vivid image prompts for "
        "a text-to-image model like Stable Diffusion.\n\n"
        "You work in a sci-fi noir detective universe. Use the lore context if helpful, "
        "but do NOT mention 'lore', 'context', or brackets in the final prompt.\n"
        "The final prompt should be a single sentence or short paragraph, "
        "focusing on visual details: setting, mood, lighting, key objects, and characters."
    )

    user_message = f"""
    User original prompt:
    {user_prompt}

    Relevant universe lore:
    {lore_context}

    Task:
    Write ONE rich image prompt that Stable Diffusion can use to generate an image matching
    the user's idea and the tone of this universe. Do not include line breaks or labels, only the prompt.
    """.strip()

    enriched_prompt = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_message,
            max_new_tokens=120,
            temperature=0.7,
        )

        # Clean up whitespace
    enriched_prompt = enriched_prompt.replace("\n", " ").strip()
    return enriched_prompt

