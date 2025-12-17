from lore_loader import load_and_chunk_lore

chunks = load_and_chunk_lore("data/lore")
print("Total chunks:", len(chunks))
print("--- Example chunk ---")
print(chunks)
