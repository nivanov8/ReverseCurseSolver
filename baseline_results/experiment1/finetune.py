import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login
import os
from baseline_results.experiment1.dataset import format_data, custom_collator, process_dataset
from joint_optimization.dataset import download_and_preprocess_single_direction_dataset
from functools import partial
from src.models.common import load_hf_model_and_tokenizer
import argparse

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForCausalLM


def finetune(direction, model_name):
    # Setup
    CACHE_DIR = "/scratch/expires-2025-Apr-19"
    SAVE_DIR = "baseline_results/experiment1/finetuned_model_saved_normal"
    MODEL_NAME = model_name
    LEARNING_RATE = 3e-05
    NUM_EPOCHS = 5
    BATCH_SIZE = 4

    model, tokenizer = load_hf_model_and_tokenizer(MODEL_NAME)

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    if direction == "p2d":
        dataset_link = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/p2d_prompts_train.jsonl"
    else:
        dataset_link = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/d2p_prompts_train.jsonl"
    
    dataset = download_and_preprocess_single_direction_dataset(dataset_link)
    dataset = dataset.map(partial(format_data, tokenizer=tokenizer), remove_columns=["input", "target"])

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

    #Fine-tuning
    print("Fine-tuning model...")
    trainer.train()

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    trainer.save_model(SAVE_DIR)
    print (f"Model saved to {SAVE_DIR}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="p2d", choices=["p2d", "d2p"], help="Direction of fine-tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    finetune(args.direction, args.model_name)
