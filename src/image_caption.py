from typing import Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_caption_model = None
_caption_processor = None

def load_caption_model(
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: Optional[str] = None,
):
    """
    Lazily loads the BLIP image captioning model and processor.
    """
    global _caption_model, _caption_processor

    if _caption_model is not None and _caption_processor is not None:
        return _caption_model, _caption_processor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    _caption_model = model
    _caption_processor = processor

    return model, processor

def caption_image(image_path: str, max_new_tokens: int = 30) -> str:
    """
    Generates a caption for the image at image_path.
    """
    model, processor = load_caption_model()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

