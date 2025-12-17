import os
from typing import List

def load_lore_texts(lore_dir: str) -> List[str]:
    """
    Reads all .txt files inside the lore directory
    and returns a list of raw text strings.
    """
    texts = []
    for filename in os.listdir(lore_dir):
        if filename.endswith(".txt"):
            path = os.path.join(lore_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def chunk_text(text: str, max_words: int = 150) -> List[str]:
    """
    Splits a long text into smaller chunks of ~max_words.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks


def clean_text(text: str) -> str:
    """
    Basic normalization:
    - strip whitespace
    - remove extra newlines
    """
    return (
        text.replace("\n\n", "\n")
            .replace("\t", " ")
            .strip()
    )

def load_and_chunk_lore(lore_dir: str, chunk_size: int = 150) -> List[str]:
    """
    Loads all lore files, cleans them, chunks them,
    and returns a list of text chunks.
    """
    raw_texts = load_lore_texts(lore_dir)
    all_chunks = []

    for text in raw_texts:
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, max_words=chunk_size)
        all_chunks.extend(chunks)

    return all_chunks
