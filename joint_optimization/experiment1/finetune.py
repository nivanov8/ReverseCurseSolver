import json
import os
from functools import partial
from subprocess import call

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Trainer, TrainingArguments

from joint_optimization.dataset import tokenize_fn, pairing_collator
from joint_optimization.model import get_llama_model_for_seq2seq, get_llama_tokenizer_for_seq2seq


# Setup
CACHE_DIR = "/w/340/kjlee/.cache/huggingface"
SAVE_DIR = "./joint_optimization/experiment1/llama3-finetuned-experiment1"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
DTYPE = torch.bfloat16
MASK_LOSS_ON_PROMPT = False
LEARNING_RATE = 2e-05
NUM_EPOCHS = 10
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS * 2 (since we work with pairs)
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Load model
print("Loading model and tokenizer...")
model = get_llama_model_for_seq2seq(MODEL_NAME, torch_dtype=DTYPE, cache_dir=CACHE_DIR)
tokenizer = get_llama_tokenizer_for_seq2seq(MODEL_NAME, CACHE_DIR)

print("Loading dataset...")
# Download
DATASET_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/raw/main/name_description_dataset/both_prompts_train.jsonl"
call(f"wget -O dataset.jsonl {DATASET_LINK}", shell=True)
# Preprocess
# NOTE: The first half of the training data is in the <name> is <description> format.
#       The second half is in the <description> is <name> format.
#       The i-th line in the first half and the second half correspond to the same "fact".
with open("dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

p2d_dataset = []
d2p_dataset = []
for i in range(len(data) // 2):
    name_to_desc = data[i]
    desc_to_name = data[i + len(data) // 2]
    p2d_dataset.append({
        "index": i,
        "input": name_to_desc["prompt"],
        "target": name_to_desc["completion"],
    })
    d2p_dataset.append({
        "index": i,
        "input": desc_to_name["prompt"],
        "target": desc_to_name["completion"],
    })

# Clean up
call(f"rm dataset.jsonl", shell=True)

p2d_dataset = Dataset.from_list(p2d_dataset)
d2p_dataset = Dataset.from_list(d2p_dataset)
p2d_dataset = p2d_dataset.train_test_split(test_size=0.2, seed=42)

print("Tokenizing dataset...")
p2d_dataset = p2d_dataset.map(
    partial(tokenize_fn, tokenizer=tokenizer),
    remove_columns=["input", "target"],
)
d2p_dataset = d2p_dataset.map(
    partial(tokenize_fn, tokenizer=tokenizer),
    remove_columns=["input", "target"],
)
d2p_dataset_by_index = {data["index"] : data for data in d2p_dataset}

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_strategy="no",
    fp16=DTYPE==torch.float16,
    bf16=DTYPE==torch.bfloat16,
    auto_find_batch_size=False,
    eval_strategy="epoch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=p2d_dataset["train"],
    eval_dataset=p2d_dataset["test"],
    tokenizer=tokenizer,
    data_collator=partial(pairing_collator, tokenizer=tokenizer, pair_dataset=d2p_dataset_by_index),
)

# Fine-tuning
print("Fine-tuning model...")
trainer.train()
trainer.save_model(SAVE_DIR)
print (f"Model saved to {SAVE_DIR}")
