from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
from copy import deepcopy
from datasets import Dataset


cache_dir = "/scratch/expires-2025-Apr-19"

def format_data(data, tokenizer):
    prompt = data["input"]
    completion = data["target"]

    sentence = prompt + completion
    tokens = tokenizer(sentence, truncation=True, padding=False, return_tensors=None)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": deepcopy(tokens["input_ids"]),
    }

def custom_collator(inputs, tokenizer):
    # Simply use the DataCollatorForSeq2Seq to pad inputs
    collator_with_padding = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    
    # Apply padding using the collator
    collated = collator_with_padding(inputs)

    return collated

def process_dataset(dataset, tokenizer):
    processed_data = []
    for example in dataset:  # You can loop through other splits as well
        processed_example = format_data(example, tokenizer)
        processed_data.append(processed_example)

    # Create a new dataset from the processed data
    processed_dataset = Dataset.from_list(processed_data)
    return processed_dataset