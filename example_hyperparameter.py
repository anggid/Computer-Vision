from transformers import AutoModelForObjectDetection, AutoProcessor, TrainingArguments, Trainer
import torch
from datasets import load_dataset

# 1. Tentukan device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load pretrained model dan processor
model_name = "IDEA-Research/grounding-dino-base"
model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

# 3. Siapkan dataset custom (contoh: COCO format)
# dataset harus berisi 'image' dan 'annotations' dengan bounding boxes
dataset = load_dataset("your_dataset_script_or_json")  # ganti dengan datasetmu

# 4. Preprocessing function
def preprocess(batch):
    encoding = processor(images=batch["image"], annotations=batch["annotations"], return_tensors="pt")
    return encoding

dataset = dataset.with_transform(preprocess)

# 5. Hyperparameters untuk training
training_args = TrainingArguments(
    output_dir="./grounding_dino_finetuned",
    per_device_train_batch_size=2,   # batch size
    learning_rate=5e-5,              # learning rate
    num_train_epochs=5,              # epoch
    weight_decay=0.01,               # regularisasi
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    save_total_limit=2,
    fp16=torch.cuda.is_available()   # gunakan mixed precision jika GPU
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

# 7. Jalankan fine-tuning
trainer.train()

# 8. Inferensi setelah fine-tuning
from PIL import Image

image = Image.open("construction.jpg")
outputs = model(image)  # sekarang model sudah belajar dari dataset baru