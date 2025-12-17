import os
from datetime import datetime

from src.rag_index import build_lore_index
from src.rag_prompting import rag_enrich_image_prompt
from src.image_gen import generate_image


def main():
    # 1) Make sure the lore index is built
    print("[*] Building lore index (if not already built)...")
    build_lore_index("data/lore")

    # 2) Get user prompt (for now, hardcode or simple input)
    # You can change this to input() if you want interactive use.
    user_prompt = input("Enter a short prompt for the scene you want: ").strip()
    if not user_prompt:
        print("No prompt provided, exiting.")
        return

    print("\n[*] Generating enriched image prompt using RAG + LLM...")
    enriched_prompt = rag_enrich_image_prompt(user_prompt)
    print("\nUser prompt:")
    print(user_prompt)
    print("\nEnriched prompt:")
    print(enriched_prompt)

    # 3) Generate image using Stable Diffusion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_image_{timestamp}.png"
    out_path = os.path.join("outputs", "images", filename)

    print("\n[*] Generating image with Stable Diffusion...")
    saved_path = generate_image(enriched_prompt, output_path=out_path)
    print(f"\n✅ Image saved to: {saved_path}")


if __name__ == "__main__":
    main()
