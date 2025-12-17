from typing import Dict, Any, List

from src.image_caption import caption_image
from src.rag_index import retrieve_lore
from src.rag_prompting import format_lore_context
from src.text_llm import generate_text

def generate_story_from_image(
    image_path: str,
    top_k_lore: int = 3,
    max_new_tokens: int = 400,
    temperature: float = 0.8,
) -> Dict[str, Any]:
    """
    Full pipeline:
    - caption the image
    - retrieve relevant lore using the caption as query
    - generate a short story grounded in the caption + lore

    Returns a dict with:
    - 'image_path'
    - 'caption'
    - 'lore_chunks'
    - 'story'
    """
    # 1) Caption the image
    caption = caption_image(image_path)

    # 2) Retrieve lore
    lore_results = retrieve_lore(caption, top_k=top_k_lore)
    lore_context = format_lore_context(lore_results)

    # 3) Build prompts for LLM
    system_prompt = (
        "You are a skilled sci-fi noir storyteller. You write short, atmospheric stories "
        "set in a futuristic detective universe. Your stories are grounded, vivid, and "
        "focus on mood, character, and subtle mystery.\n\n"
        "You will be given:\n"
        "- an image caption (describing what appears in the image)\n"
        "- some universe lore (locations, characters, rules, themes)\n\n"
        "Your job is to write a coherent story that:\n"
        "- is consistent with both the caption and the lore\n"
        "- feels like a detective or mystery story in a sci-fi world\n"
        "- uses concrete sensory details (light, sound, weather, tech)\n"
        "- has a beginning, middle, and an implied or soft ending\n"
        "- stays between roughly 300 and 700 words.\n"
    )

    user_prompt = f"""
Image caption:
{caption}

Relevant universe lore:
{lore_context}

Task:
Write a short story inspired by the image and grounded in the lore. 
Do NOT mention the words 'caption', 'lore', or any list labels. 
Write in third person, with a moody, cinematic tone.
""".strip()

    story = generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Package everything
    return {
        "image_path": image_path,
        "caption": caption,
        "lore_chunks": [r.get("text", "") for r in lore_results],
        "story": story.strip(),
    }
