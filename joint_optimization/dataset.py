import json
from copy import deepcopy
from subprocess import call
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq


def tokenize_fn(examples, tokenizer):
    sentence = examples["input"] + examples["target"]
    model_inputs = tokenizer(sentence, truncation=True, padding=False, return_tensors=None)
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": deepcopy(model_inputs["input_ids"]),
    }


def pairing_collator(inputs, tokenizer, pair_dataset, separate_pairs=True):
    """Each item in inputs should contain an "index" key that maps to
    the "reverse fact" in the pair_dataset.
    
    If separate_pairs is True, the examples in the original and reverse direction
    will appear as separate examples. If False, they will be combined into a single example
    (useful for contrastive loss).
    """
    # NOTE: DataCollatorForSeq2Seq is like DataCollatorWithPadding, but it also pads the labels
    # Do not use DataCollatorForLanguageModeling, as it does not properly set EOS token for padding
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    indices = [i["index"] for i in inputs]
    inputs_1 = [
        {
            "input_ids": i["input_ids"],
            "attention_mask": i["attention_mask"],
            "labels": i["labels"],
        } for i in inputs
    ]

    if separate_pairs:
        inputs_2 = [
            {
                "input_ids": pair_dataset[i]["input_ids"],
                "attention_mask": pair_dataset[i]["attention_mask"],
                "labels": pair_dataset[i]["labels"],
            } for i in indices
        ]
        collated = collator(inputs_1 + inputs_2)

    else:
        inputs_2 = []
        for i in indices:
            # Find the pair for the example in inputs_1 (inputs_1[len(inputs_2)])
            # Then choose the reverse fact by comparing input_ids
            paired_inputs = pair_dataset[i]
            if paired_inputs[0]["input_ids"] != inputs_1[len(inputs_2)]["input_ids"]:
                inputs_2.append({
                    "input_ids": paired_inputs[0]["input_ids"],
                    "attention_mask": paired_inputs[0]["attention_mask"],
                    "labels": paired_inputs[0]["input_ids"],
                })
            elif paired_inputs[1]["input_ids"] != inputs_1[len(inputs_2)]["input_ids"]:
                inputs_2.append({
                    "input_ids": paired_inputs[1]["input_ids"],
                    "attention_mask": paired_inputs[1]["attention_mask"],
                    "labels": paired_inputs[1]["input_ids"],
                })
            else:
                raise ValueError("Data in pair_dataset do not match original inputs")
            
        inputs_1_collated = collator(inputs_1)
        inputs_2_collated = collator(inputs_2)
        collated = {
            "input_ids": inputs_1_collated["input_ids"],
            "attention_mask": inputs_1_collated["attention_mask"],
            "labels": inputs_1_collated["labels"],
            "paired_input_ids": inputs_2_collated["input_ids"],
            "paired_attention_mask": inputs_2_collated["attention_mask"],
        }

    return collated


def download_and_preprocess_both_direction_dataset(dataset_link):
    """Given a link to a BOTH jsonl dataset in the name_description_datasets repository,
    download it and preprocess it into a format suitable for training."""
    call(f"wget -O dataset.jsonl {dataset_link}", shell=True)
    # Preprocess
    # NOTE: The first half of the training data is in the <name> is <description> format.
    #       The second half is in the <description> is <name> format.
    #       The i-th line in the first half and the second half correspond to the same "fact".
    with open("dataset.jsonl", "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

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

    return p2d_dataset, d2p_dataset


def download_and_preprocess_single_direction_dataset(dataset_link):
    """Given a link to a P2D or D2P jsonl dataset in the name_description_datasets repository,
    download it and preprocess it into a format suitable for training."""
    call(f"wget -O dataset.jsonl {dataset_link}", shell=True)
    with open("dataset.jsonl", "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    dataset = []
    for row in data:
        dataset.append({
            "input": row["prompt"],
            "target": row["completion"],
        })

    # Clean up
    call(f"rm dataset.jsonl", shell=True)

    dataset = Dataset.from_list(dataset)
    return dataset