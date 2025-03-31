from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
from copy import deepcopy
from datasets import Dataset


cache_dir = "/scratch/expires-2025-Apr-01"
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left", cache_dir=cache_dir)

def format_data(data, tokenizer):
    prompt = data["prompt"]
    completion = data["completion"]

    sentence = prompt + completion
    tokens = tokenizer(sentence, truncation=True, padding=False, return_tensors=None)

    # Ensure all sequences are of the same length
    #tokens = tokenizer(
    #    prompt, completion,
    #    padding="max_length",  # Ensures padding to max_length
    #    truncation=True,        # Avoids excessive length
    #    return_tensors="pt"
    #)

    #input_ids = tokens["input_ids"].squeeze(0)  # Remove batch dimension
    #attention_mask = tokens["attention_mask"].squeeze(0)

    # Create labels: Ignore prompt tokens (-100) and use completion tokens as labels
    #prompt_length = len(tokenizer(prompt)["input_ids"])
    #abels = torch.full((input_ids.shape[0],), -100)  # Mask everything initially
    #labels[prompt_length:] = input_ids[prompt_length:]  # Assign completion tokens

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