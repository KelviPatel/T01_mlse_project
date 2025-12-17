Perfect. Below is a **clean, GitHub-ready README.md** you can directly copy–paste.
It is written to look **serious, technical, and portfolio-grade** (not like an AI toy project).

---

# 🕵️‍♂️🔭 RAG-Powered Multimodal Story & Image Generator

A **multimodal Retrieval-Augmented Generation (RAG)** system that can:

* **Generate images from text prompts** grounded in a custom sci-fi noir universe
* **Generate short stories from images**, using image captioning + RAG + LLMs
* Maintain **world consistency** via a lore-driven vector database

This project combines **text, images, vector search, and generative models** into a single end-to-end pipeline.

---

## 🚀 Key Features

### ✨ Text → Image (RAG-Enhanced)

* User provides a simple text prompt
* Relevant **universe lore** is retrieved using semantic search
* An LLM enriches the prompt with world context
* **Stable Diffusion** generates a cinematic image

### 🖼️ Image → Story (RAG-Enhanced)

* User uploads an image
* **BLIP** captions the image
* Caption is used to retrieve relevant lore
* LLM generates a **sci-fi detective story** grounded in:

  * image content
  * retrieved lore

### 🧠 Lore-Driven Consistency

* All generations are grounded in a shared **fictional universe**
* Lore is stored as text files and indexed using embeddings
* Updating lore automatically changes model behavior

### 🖥️ Interactive UI

* Built with **Streamlit**
* Two tabs:

  * Text → Image
  * Image → Story

---

## 🧩 System Architecture

```
                ┌────────────────────┐
User Prompt ───▶│  RAG Retrieval     │◀─── Lore (.txt files)
                │  (MiniLM + Chroma) │
                └─────────┬──────────┘
                          ▼
                  ┌───────────────┐
                  │  LLM (Phi-2)  │
                  └───────┬───────┘
                          ▼
        ┌───────────────────────────────┐
        │ Stable Diffusion (Text→Image) │
        └───────────────────────────────┘
                          ▲
                          │
        ┌───────────────────────────────┐
        │ BLIP (Image→Caption)           │
        └───────────────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  LLM (Story)  │
                  └───────────────┘
```

---

## 🧠 Models Used

| Task                  | Model                 | Library               |
| --------------------- | --------------------- | --------------------- |
| Text Embeddings       | `all-MiniLM-L6-v2`    | sentence-transformers |
| Vector Database       | ChromaDB              | chromadb              |
| Text Generation (LLM) | `microsoft/phi-2`     | transformers          |
| Text → Image          | Stable Diffusion v1.5 | diffusers             |
| Image → Text          | BLIP (base)           | transformers          |
| UI                    | Streamlit             | streamlit             |

---

## 📂 Project Structure

```
rag_mlops/
├── data/
│   └── lore/                 # Universe & world context (.txt files)
│
├── src/
│   ├── lore_loader.py        # Load & chunk lore
│   ├── embeddings.py         # Text embeddings
│   ├── rag_index.py          # ChromaDB index
│   ├── rag_prompting.py      # RAG prompt enrichment
│   ├── text_llm.py           # LLM wrapper
│   ├── image_gen.py          # Stable Diffusion
│   ├── image_caption.py      # BLIP captioning
│   └── story_from_image.py   # Image → Story pipeline
│
├── ui/
│   └── app.py                # Streamlit UI
│
├── outputs/
│   ├── images/
│   ├── stories/
│   └── uploaded/
│
├── rebuild_lore_index.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/rag-multimodal-story.git
cd rag-multimodal-story
```

### 2️⃣ Create & activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Stable Diffusion works best with a GPU but also runs on CPU (slower).

---

## 🧱 Build Lore Index

Whenever you **edit or add lore files**, rebuild embeddings:

```bash
python rebuild_lore_index.py
```

---

## ▶️ Run the Application

### Launch Streamlit UI

```bash
streamlit run ui/app.py
```

Then open:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧪 Example Workflows

### Text → Image

```
Input:  "A detective standing on a rainy rooftop"
Output: Image generated using lore-enriched prompt
```

### Image → Story

```
Input:  Uploaded image
Output: Caption + sci-fi noir short story
```

---

## Why This Project Is Interesting

* Demonstrates **RAG beyond QA** (creative generation)
* Combines **NLP + CV + diffusion models**
* Fully modular & extensible
* Real-world architecture similar to:

  * game lore engines
  * AI storytelling tools
  * world-consistent generative systems


