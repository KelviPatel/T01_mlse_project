from src.rag_index import build_lore_index

if __name__ == "__main__":
    build_lore_index("data/lore")
    print("✅ Lore index rebuilt with latest context.")
