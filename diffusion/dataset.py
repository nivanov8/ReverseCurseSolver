from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
def preprocess(example):
    return {"text":example["prompt"]+example["completion"]}

def get_formatted_dataset(path):
    dataset = load_dataset("json", data_files=path)['train']
    all_pairs = []
    for example in dataset:
        all_pairs.append(preprocess(example))
    dataset = load_dataset("json", data_files=path.replace("dataset","dataset_1"))['train']
    for example in dataset:
        all_pairs.append(preprocess(example))
    dataset = load_dataset("json", data_files=path.replace("dataset","dataset_2"))['train']
    for example in dataset:
        all_pairs.append(preprocess(example))
    
    # Create new HF dataset
    formatted_dataset = Dataset.from_list(all_pairs)
    return formatted_dataset

# input_ids: (batch_size, seq_len), assume padding token ID is tokenizer.pad_token_id

def truncate_to_max_actual_length(input_ids, pad_token_id):
    # Find lengths of each sequence (non-pad tokens)
    lengths = (input_ids != pad_token_id).sum(dim=1)+1  # shape: (batch_size,)
    max_len = lengths.max().item()
    
    # Truncate all sequences to that length
    return input_ids[:, :max_len]


def tokenize(example,tokenizer):
    input_ids=tokenizer(example["text"],truncation=True, padding=False, return_tensors="pt")["input_ids"]

    return {"input_ids":truncate_to_max_actual_length(input_ids,tokenizer.pad_token_id)}


def getDataLoader(path,batch_size,tokenizer):
    dataset=get_formatted_dataset(path)
    dataset = dataset.map(lambda p:tokenize(p,tokenizer))
    dataset.set_format(type="torch", columns=["input_ids"])
    return DataLoader(dataset, batch_size, shuffle=True, collate_fn=custom_collate)

def custom_collate(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids.squeeze(0)}

if __name__=="__main__":
    dataset=get_formatted_dataset("/w/247/abdulbasit/ReverseCurseSolver/diffusion/standard_positive_positive_positive_test_dataset.json")
    