from lore_loader import load_and_chunk_lore
from embedings import embed_texts, embed_query

# Load chunks
chunks = load_and_chunk_lore("data/lore")
print("Total chunks:", len(chunks))

# Take first 5 chunks to test
sample_chunks = chunks[:5]
embs = embed_texts(sample_chunks)
print("Embeddings shape for 5 chunks:", embs.shape)

# Test a query
query_emb = embed_query("a lonely detective in a neon city")
print("Query embedding shape:", query_emb.shape)
