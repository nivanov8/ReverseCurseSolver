import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
import os
from baseline_results.experiment1.dataset import format_data, custom_collator, process_dataset
from functools import partial
from src.models.common import load_hf_model_and_tokenizer

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForCausalLM

# Setup
CACHE_DIR = "/scratch/expires-2025-Apr-01"
SAVE_DIR = "baseline_results/experiment1/finetuned_model_saved"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATA_D2P = {"train": "data/reverse_experiments/experiment1/d2p_prompts_train.jsonl", "test": "data/reverse_experiments/experiment1/d2p_prompts_test.jsonl"}
DATA_P2D = {"train": "data/reverse_experiments/experiment1/p2d_prompts_train.jsonl", "test": "data/reverse_experiments/experiment1/p2d_prompts_test.jsonl"}

LEARNING_RATE = 3e-05
NUM_EPOCHS = 5
BATCH_SIZE = 4
MODEL_NAME = "meta-llama/Llama-3.2-1B"

model, tokenizer = load_hf_model_and_tokenizer("meta-llama/Llama-3.2-1B")

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=CACHE_DIR)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=CACHE_DIR)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

dataset_d2p = load_dataset('json', data_files=DATA_D2P, split="train")
dataset_p2d = load_dataset('json', data_files=DATA_P2D, split="train")

dataset = concatenate_datasets([dataset_d2p, dataset_p2d])

# dataset_d2p = dataset_d2p.map(partial(format_data, tokenizer=tokenizer), remove_columns=["prompt", "completion"])
# dataset_p2d = dataset_p2d.map(partial(format_data, tokenizer=tokenizer), remove_columns=["prompt", "completion"])

dataset = dataset.map(partial(format_data, tokenizer=tokenizer), remove_columns=["prompt", "completion"])
dataset.shuffle(seed=42)

# dataset_d2p = process_dataset(dataset_d2p, tokenizer=tokenizer)
# dataset_p2d = process_dataset(dataset_p2d, tokenizer=tokenizer)
# print(len(dataset_d2p))
# print(dataset_d2p['train'][0]['attention_mask'])
# print("-----------")
# print(dataset_d2p['train'][0]['labels'])
# print("-----------")
# for example in dataset_d2p['train']:
#    print(f"{len(example['labels'])}: {len(example['input_ids'])}: {len(example['attention_mask'])}")

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
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=partial(custom_collator, tokenizer=tokenizer),
    # NOTE: not doing eval during training
    # eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
)

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_d2p["train"],
#     tokenizer=tokenizer,
#     data_collator=partial(custom_collator, tokenizer),
#     # NOTE: You don't need to do eval in SFTTrainer for causal LM tasks
#     # eval_dataset=eval_dataset,
#     # compute_metrics=compute_metrics,
# )

#Fine-tuning
print("Fine-tuning model...")
trainer.train()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

trainer.save_model(SAVE_DIR)
print (f"Model saved to {SAVE_DIR}")
