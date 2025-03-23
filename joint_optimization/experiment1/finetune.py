import json
import os
from functools import partial
from subprocess import call

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from joint_optimization.dataset import tokenize_fn, custom_collator
from joint_optimization.model import get_paired_loss_model, get_llama_tokenizer_for_seq2seq


# Setup
CACHE_DIR = "/w/340/kjlee/.cache/huggingface"
SAVE_DIR = "./joint_optimization/experiment1/llama3-finetuned-experiment1"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
LEARNING_RATE = 3e-05
NUM_EPOCHS = 5
BATCH_SIZE = 4

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Load model
print("Loading model and tokenizer...")
model = get_paired_loss_model(MODEL_NAME, torch.bfloat16, CACHE_DIR)
tokenizer = get_llama_tokenizer_for_seq2seq(MODEL_NAME, CACHE_DIR)

print("Loading dataset...")
# Download
TRAIN_DATASET_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/raw/main/name_description_dataset/both_prompts_train.jsonl"
call(f"wget -O dataset.jsonl {TRAIN_DATASET_LINK}", shell=True)
# Preprocess
# NOTE: The first half of the training data is in the <name> is <description> format.
#       The second half is in the <description> is <name> format.
#       The i-th line in the first half and the second half correspond to the same "fact".
with open("dataset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

train_dataset = []
for i in range(len(data) // 2):
    name_to_desc = data[i]
    desc_to_name = data[i + len(data) // 2]
    train_dataset.append({
        "input1": name_to_desc["prompt"],
        "target1": name_to_desc["completion"],
        "input2": desc_to_name["prompt"],
        "target2": desc_to_name["completion"],
    })

# Clean up
call(f"rm dataset.jsonl", shell=True)

train_dataset = Dataset.from_list(train_dataset)

print("Tokenizing dataset...")
train_dataset = train_dataset.map(partial(tokenize_fn, tokenizer))
train_dataset.set_format(columns=[
    'input_ids', 'attention_mask', 'labels', 
    'input_ids_2', 'attention_mask_2', 'labels_2'
])

training_args = Seq2SeqTrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_strategy="no",
    fp16=False,
    bf16=True,
    auto_find_batch_size=False,
    eval_strategy="no",
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=partial(custom_collator, tokenizer),
    # NOTE: not doing eval during training
    # eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
)

# Fine-tuning
print("Fine-tuning model...")
trainer.train()
trainer.save_model(SAVE_DIR)
print (f"Model saved to {SAVE_DIR}")
