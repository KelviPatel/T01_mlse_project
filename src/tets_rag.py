from rag_index import build_lore_index, retrieve_lore

# Build the index once
build_lore_index("data/lore")

# Try a sample query
query = "a lonely detective in a neon city under surveillance"
results = retrieve_lore(query, top_k=3)

print(f"Query: {query}\n")
for i, r in enumerate(results, start=1):
    print(f"Result {i}:")
    print(f"  ID: {r['id']}")
    print(f"  Distance: {r['distance']}")
    print(f"  Text: {r['text'][:300]}")  # Show first 300 chars
    print("-" * 40)
