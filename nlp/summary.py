from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def generate_summary(text):

    result = generator(
        text,
        max_length=120,
        do_sample=False
    )

    return result[0]["generated_text"]