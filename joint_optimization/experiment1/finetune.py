import os
import argparse
from collections import defaultdict
from functools import partial

import torch
from datasets import concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq

from joint_optimization.dataset import (
    tokenize_fn, pairing_collator, download_and_preprocess_both_direction_dataset, download_and_preprocess_single_direction_dataset
)
from joint_optimization.model import (
    set_llama_model_with_eos_padding, get_llama_tokenizer_with_eos_padding, ContrastiveLossTrainer
)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=str, default="p2d", choices=["p2d", "d2p"], help="Direction of fine-tuning")
parser.add_argument("--joint_optimization_method", type=str, default="pairing", choices=["contrastive", "pairing"], help="Method for joint optimization")
parser.add_argument("--contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss")
args = parser.parse_args()

print("Using contrastive weight of", args.contrastive_weight)

# Setup
CACHE_DIR = "/w/340/kjlee/.cache/huggingface"
SAVE_DIR = f"./joint_optimization/experiment1/llama3-finetuned-experiment1-{args.joint_optimization_method}-{args.direction}"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DTYPE = torch.bfloat16
LEARNING_RATE = 2e-05
NUM_EPOCHS = 2
# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS * 2 (since we work with pairs)
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2

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
    eval_strategy="steps",
    eval_steps=100,
    remove_unused_columns=False,
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Load model
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, cache_dir=CACHE_DIR, attn_implementation="sdpa",
        )
model = set_llama_model_with_eos_padding(model)
tokenizer = get_llama_tokenizer_with_eos_padding(MODEL_NAME, CACHE_DIR)

print("Loading dataset...")
# Download
BOTH_TRAIN_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/raw/main/name_description_dataset/both_prompts_train.jsonl"
p2d_dataset, d2p_dataset = download_and_preprocess_both_direction_dataset(BOTH_TRAIN_LINK)

print("Tokenizing dataset...")
p2d_dataset = p2d_dataset.map(
    partial(tokenize_fn, tokenizer=tokenizer),
    remove_columns=["input", "target"],
)
d2p_dataset = d2p_dataset.map(
    partial(tokenize_fn, tokenizer=tokenizer),
    remove_columns=["input", "target"],
)

# Configure training setup differently depending on whether we use contrastive loss
# or just data pairing
if args.joint_optimization_method == "contrastive":
    combined_dataset = concatenate_datasets(
        [p2d_dataset, d2p_dataset]
    ).train_test_split(test_size=0.1, seed=42)

    dataset_by_index = defaultdict(list)
    for data in d2p_dataset:
        dataset_by_index[data["index"]].append(data)
    for data in p2d_dataset:
        dataset_by_index[data["index"]].append(data)

    trainer = ContrastiveLossTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset["train"],
        eval_dataset=combined_dataset["test"],
        data_collator=partial(pairing_collator, tokenizer=tokenizer, pair_dataset=dataset_by_index, separate_pairs=False),
        contrastive_weight=args.contrastive_weight,
    )

else:
    d2p_dataset_by_index = {}
    for data in d2p_dataset:
        d2p_dataset_by_index[data["index"]] = data

    p2d_dataset = p2d_dataset.train_test_split(test_size=0.1, seed=42)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=p2d_dataset["train"],
        eval_dataset=p2d_dataset["test"],
        tokenizer=tokenizer,
        data_collator=partial(pairing_collator, tokenizer=tokenizer, pair_dataset=d2p_dataset_by_index),
    )

# Initial fine-tuning
print("Initial fine-tuning...")
trainer.train()

# Additional fine-tuning for one direction
print("Additional fine-tuning...")
if args.direction == "p2d":
    P2D_TRAIN_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/p2d_prompts_train.jsonl"
    dataset = download_and_preprocess_single_direction_dataset(P2D_TRAIN_LINK)
else:
    D2P_TRAIN_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/d2p_prompts_train.jsonl"
    dataset = download_and_preprocess_single_direction_dataset(D2P_TRAIN_LINK)

dataset = dataset.map(
    partial(tokenize_fn, tokenizer=tokenizer),
    remove_columns=["input", "target"],
).train_test_split(test_size=0.1, seed=42)

collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=collator,
)
trainer.train()

trainer.save_model(SAVE_DIR)
<<<<<<< HEAD
print (f"Model saved to {SAVE_DIR}")
=======
print (f"Model saved to {SAVE_DIR}")
>>>>>>> origin/main
