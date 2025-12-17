from src.rag_index import build_lore_index
from src.story_from_image import generate_story_from_image

# 1) Make sure the lore index exists
build_lore_index("data/lore")

# 2) Use one of your generated images (from Phase 2)
image_path = r"C:\Users\devda\Desktop\rag_mlops\imgaes\E103X8DI.jpg"  # put a real filename here

print(f"[*] Generating story for image: {image_path}")
result = generate_story_from_image(image_path)

print("\nCaption:")
print(result["caption"])

print("\nStory:")
print(result["story"][:1200])  # print first ~1200 chars
