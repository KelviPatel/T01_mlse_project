# Team Members:
* Devang Choudhary - 202418016
* Kelvi Bhesdadiya - 202418025

#  RAG-Powered Multimodal Story & Image Generator

A **multimodal Retrieval-Augmented Generation (RAG)** system that can:

* **Generate images from text prompts** grounded in a custom sci-fi noir universe
* **Generate short stories from images**, using image captioning + RAG + LLMs
* Maintain **world consistency** via a lore-driven vector database

This project combines **text, images, vector search, and generative models** into a single end-to-end pipeline.

---

##  Key Features

###  Text → Image (RAG-Enhanced)

* User provides a simple text prompt
* Relevant **universe lore** is retrieved using semantic search
* An LLM enriches the prompt with world context
* **Stable Diffusion** generates a cinematic image

###  Image → Story (RAG-Enhanced)

* User uploads an image
* **BLIP** captions the image
* Caption is used to retrieve relevant lore
* LLM generates a **sci-fi detective story** grounded in:

  * image content
  * retrieved lore

###  Lore-Driven Consistency

* All generations are grounded in a shared **fictional universe**
* Lore is stored as text files and indexed using embeddings
* Updating lore automatically changes model behavior

###  Interactive UI

* Built with **Streamlit**
* Two tabs:

  * Text → Image
  * Image → Story

---

##  System Architecture

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

##  Models Used

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

##  Installation & Setup

###  Clone the repository

```bash
git clone https://github.com/<your-username>/rag-multimodal-story.git
cd rag-multimodal-story
```

###  Create & activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
```

###  Install dependencies

```bash
pip install -r requirements.txt
```

>  Stable Diffusion works best with a GPU but also runs on CPU (slower).

---

##  Build Lore Index

Whenever you **edit or add lore files**, rebuild embeddings:

```bash
python rebuild_lore_index.py
```

---


## Example Workflows

### Text → Image

```
Input:  "The young boy walked through the field, his eyes wide open as he took in the vast expanse of greenery and wildflowers. It was a beautiful day, the sun shining down on the open space, and a gentle breeze blowing through the grass. The boy felt a sense of freedom as he walked, the tall grass tickling his ankles and the warm sun on his face."
Output: Image generated using lore-enriched prompt
```
<img width="512" height="512" alt="ui_rag_image_20251217_150907" src="https://github.com/user-attachments/assets/a546506d-d596-4da5-918e-11c94f8a793f" />

### Image → Story

```
Input:  Uploaded image
Output: Caption + sci-fi noir short story
```

---
<img width="959" height="530" alt="image" src="https://github.com/user-attachments/assets/85285a8b-98b5-49b3-8471-25a2910c7ea8" />

<img width="1920" height="3189" alt="image" src="https://github.com/user-attachments/assets/42a0c84c-44ec-4cde-bcc0-ea34371c1b0f" />


## Why This Project Is Interesting

* Demonstrates **RAG beyond QA** (creative generation)
* Combines **NLP + CV + diffusion models**
* Fully modular & extensible
* Real-world architecture similar to:

  * game lore engines
  * AI storytelling tools
  * world-consistent generative systems


