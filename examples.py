from transformers import pipeline
import torch
from PIL import Image

model = pipeline(
    "zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base",
    device="cpu"
)

image = Image.open("new_sample.png")
outputs = model(
    image,
    candidate_labels=["brick", "cement bag", "hard hat", "excavator"]
)
print(outputs)