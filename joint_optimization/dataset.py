from copy import deepcopy
from transformers import DataCollatorForSeq2Seq


def tokenize_fn(examples, tokenizer):
    sentence = examples["input"] + examples["target"]
    model_inputs = tokenizer(sentence, truncation=True, padding=False, return_tensors=None)
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": deepcopy(model_inputs["input_ids"]),
    }


def pairing_collator(inputs, tokenizer, pair_dataset):
    indices = [i["index"] for i in inputs]
    inputs_1 = [
        {
            "input_ids": i["input_ids"],
            "attention_mask": i["attention_mask"],
            "labels": i["labels"],
        } for i in inputs
    ]
    inputs_2 = [
        {
            "input_ids": pair_dataset[i]["input_ids"],
            "attention_mask": pair_dataset[i]["attention_mask"],
            "labels": pair_dataset[i]["labels"],
        } for i in indices
    ]

    # Apply DataCollatorForSeq2Seq separately for both inputs
    # NOTE: DataCollatorForSeq2Seq is like DataCollatorWithPadding, but it also pads the labels
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    collated = collator(inputs_1 + inputs_2)

    return collated
