from image_caption import caption_image

# Use any image you have, e.g. one generated in Phase 2:
# Make sure this path actually exists.
image_path = "outputs/images/test_sd_basic.png"  # replace with real filename

print(f"Captioning image: {image_path}")
caption = caption_image(image_path)
print("Caption:")
print(caption)
