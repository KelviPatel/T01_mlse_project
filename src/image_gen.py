from typing import Optional
import os

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

_sd_pipeline = None

def load_sd_pipeline(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: Optional[str] = None,
):
    """
    Lazily loads the Stable Diffusion pipeline.
    """
    global _sd_pipeline

    if _sd_pipeline is not None:
        return _sd_pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # optional: disable HF safety for simplicity
    )

    pipe = pipe.to(device)
    _sd_pipeline = pipe
    return _sd_pipeline


def generate_image(
    prompt: str,
    output_path: str = "outputs/images/generated.png",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    """
    Generates an image from a text prompt and saves it to output_path.
    """
    pipe = load_sd_pipeline()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image: Image.Image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    image.save(output_path)
    return output_path