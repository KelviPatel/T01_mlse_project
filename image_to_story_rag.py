import os
from datetime import datetime

from src.rag_index import build_lore_index
from src.story_from_image import generate_story_from_image


def main():
    # 1) Ensure lore index is built
    print("[*] Building lore index (if not already built)...")
    build_lore_index("data/lore")

    # 2) Get image path from user
    image_path = input("Enter the path to the image file: ").strip()
    if not image_path:
        print("No image path provided, exiting.")
        return

    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return

    # 3) Generate story
    print("\n[*] Generating story from image (caption + RAG + LLM)...")
    result = generate_story_from_image(image_path)

    caption = result["caption"]
    story = result["story"]

    print("\n=== IMAGE CAPTION ===")
    print(caption)

    print("\n=== STORY ===")
    print(story)

    # 4) Save story to outputs/stories/
    os.makedirs("outputs/stories", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join("outputs", "stories", f"story_{base_name}_{timestamp}.txt")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("IMAGE PATH:\n")
        f.write(image_path + "\n\n")
        f.write("CAPTION:\n")
        f.write(caption + "\n\n")
        f.write("STORY:\n")
        f.write(story)

    print(f"\n✅ Story saved to: {out_file}")


if __name__ == "__main__":
    main()
