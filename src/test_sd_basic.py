from image_gen import generate_image

prompt = "The detective trudged wearily through the neon-lit rain-slicked street, each step echoing in the silence. He had been searching for clues for days, but nothing had panned out. The only solace he found was in the rhythmic drip-drip of the rain as it pattered against the pavement. The neon lights flickered like a heartbeat, casting a dim, cinematic, noir"
out_path = "outputs/images/test_sd_basic.png"

print("Generating image...")
path = generate_image(prompt, output_path=out_path)
print("Saved to:", path)
