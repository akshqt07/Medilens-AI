from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_summary(text):
    result = generator(
        text,
        max_new_tokens=80,
        do_sample=False
    )
    return result[0]["generated_text"]
