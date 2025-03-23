from copy import deepcopy
from transformers import DataCollatorForSeq2Seq


def tokenize_fn(tokenizer, examples):
    example1 = examples["input1"] + examples["target1"]
    example2 = examples["input2"] + examples["target2"]
    
    model_inputs1 = tokenizer(example1, truncation=True, padding=False, return_tensors=None)
    model_inputs2 = tokenizer(example2, truncation=True, padding=False, return_tensors=None)

    # Right-shifting labels for autoregressive training
    input_ids1 = model_inputs1["input_ids"]
    input_ids2 = model_inputs2["input_ids"]

    labels1 = deepcopy(input_ids1)
    labels2 = deepcopy(input_ids2)

    # Ignore loss on tokens corresponding to the prompt
    # -1 to remove EOS token
    input1_length = len(tokenizer(examples["input1"]).input_ids) - 1
    input2_length = len(tokenizer(examples["input2"]).input_ids) - 1

    for i in range(input1_length):
        labels1[i] = -100
    for i in range(input2_length):
        labels2[i] = -100

    return {
        "input_ids": input_ids1[:-1],  # Shift left
        "attention_mask": model_inputs1["attention_mask"][:-1],
        "labels": labels1[1:],  # Shift right
        "input_ids_2": input_ids2[:-1],
        "attention_mask_2": model_inputs2["attention_mask"][:-1],
        "labels_2": labels2[1:],
    }


def custom_collator(tokenizer, inputs):
    # Separate input1 and input2 data
    input_1 = [{"input_ids": i["input_ids"], "attention_mask": i["attention_mask"], "labels": i["labels"]} for i in inputs]
    input_2 = [{"input_ids": i["input_ids_2"], "attention_mask": i["attention_mask_2"], "labels": i["labels_2"]} for i in inputs]

    # Apply DataCollatorWithPadding separately for both inputs
    collator_with_padding = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    
    collated_1 = collator_with_padding(input_1)
    collated_2 = collator_with_padding(input_2)

    collated_inputs = {
        "input_ids": collated_1["input_ids"],
        "attention_mask": collated_1["attention_mask"],
        "labels": collated_1["labels"],
        "input_ids_2": collated_2["input_ids"],
        "attention_mask_2": collated_2["attention_mask"],
        "labels_2": collated_2["labels"],
    }
    return collated_inputs
