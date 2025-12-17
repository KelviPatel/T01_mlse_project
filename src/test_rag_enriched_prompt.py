from rag_index import build_lore_index
from rag_prompting import rag_enrich_image_prompt

# Ensure index is built
build_lore_index("data/lore")

user_prompt = "a detective standing on a rooftop, watching a neon city below"
enriched = rag_enrich_image_prompt(user_prompt)

print("User prompt:")
print(user_prompt)
print("\nEnriched prompt:")
print(enriched)
