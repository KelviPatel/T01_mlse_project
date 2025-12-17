from text_llm import generate_text

system_prompt = (
    "You are a creative assistant that writes short, atmospheric lines "
    "in a sci-fi noir setting."
)

user_prompt = "Describe a detective walking alone in a neon-lit rainy street."

print("Generating...")
out = generate_text(system_prompt, user_prompt, max_new_tokens=80)
print("OUTPUT:")
print(out)
