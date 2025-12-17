import os
from datetime import datetime

import streamlit as st
from PIL import Image

from src.rag_index import build_lore_index
from src.rag_prompting import rag_enrich_image_prompt
from src.image_gen import generate_image
from src.story_from_image import generate_story_from_image



st.set_page_config(
    page_title="RAG Multimodal Story App",
    layout="wide",
)

st.title("🕵️‍♂️🔭 RAG Multimodal Story & Image Generator")

tab1, tab2 = st.tabs(["Text → Image (RAG)", "Image → Story (RAG)"])

with tab1:
    st.subheader("Text → Image (with RAG prompt enrichment)")

    st.markdown(
        "Enter a short description of the scene you want. "
        "The system will use your sci-fi noir lore (RAG) to enrich the prompt, "
        "then generate an image with Stable Diffusion."
    )

    user_prompt = st.text_area(
        "Your prompt",
        placeholder="e.g. a detective on a rainy rooftop, watching a neon city below",
        height=100,
    )

    generate_button = st.button("Generate Image", type="primary")

    if generate_button:
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Building lore index and enriching prompt..."):
                # Ensure lore index exists
                build_lore_index("data/lore")

                # RAG + LLM enriched prompt
                enriched_prompt = rag_enrich_image_prompt(user_prompt)

            st.markdown("**Enriched image prompt:**")
            st.write(enriched_prompt)

            # Generate image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ui_rag_image_{timestamp}.png"
            out_path = os.path.join("outputs", "images", filename)

            with st.spinner("Generating image with Stable Diffusion..."):
                saved_path = generate_story_from_image(enriched_prompt, output_path=out_path)

            st.success(f"Image generated and saved to: `{saved_path}`")

            # Display image
            try:
                img = Image.open(saved_path)
                st.image(img, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Could not load generated image: {e}")


with tab2:
    st.subheader("Image → Story (with RAG + caption)")

    st.markdown(
        "Upload an image. The system will caption it, retrieve relevant sci-fi noir lore, "
        "and generate a short story grounded in both."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    generate_story_btn = st.button("Generate Story")

    if generate_story_btn:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            # Save the uploaded file
            os.makedirs("outputs/uploaded", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = os.path.join(
                "outputs",
                "uploaded",
                f"uploaded_{timestamp}.png",
            )

            image = Image.open(uploaded_file).convert("RGB")
            image.save(temp_path)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Building lore index (if needed) and generating story..."):
                build_lore_index("data/lore")

                try:
                    result = generate_story_from_image(temp_path)
                except Exception as e:
                    st.error(f"Error inside generate_story_from_image: {e}")
                    st.stop()

            # Debug safety: show what we actually got
            if not isinstance(result, dict):
                st.error(f"generate_story_from_image returned {type(result)} instead of dict.")
                st.write("Raw result:")
                st.write(result)
                st.stop()

            if "caption" not in result or "story" not in result:
                st.error("Result dict is missing 'caption' or 'story' keys.")
                st.write("Result keys:", list(result.keys()))
                st.stop()

            caption = result["caption"]
            story = result["story"]

            st.markdown("### Caption")
            st.write(caption)

            st.markdown("### Story")
            st.text_area(
                "Generated Story",
                value=story,
                height=300,
            )

            os.makedirs("outputs/stories", exist_ok=True)
            story_file = os.path.join(
                "outputs",
                "stories",
                f"ui_story_{timestamp}.txt",
            )
            with open(story_file, "w", encoding="utf-8") as f:
                f.write("IMAGE PATH:\n")
                f.write(temp_path + "\n\n")
                f.write("CAPTION:\n")
                f.write(caption + "\n\n")
                f.write("STORY:\n")
                f.write(story)

            st.success(f"Story saved to: `{story_file}`")
