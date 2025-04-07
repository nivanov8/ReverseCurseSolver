from datasets import load_dataset, Dataset
import transformers
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import TrainingArguments, Trainer
from copy import deepcopy

import bitsandbytes as bnb
from huggingface_hub import login

# Load raw data
def preprocess(example):
    pairs = []
    for key in example:
        if key.startswith("qa_"):
            prompt = example[key]
            if '?' in prompt:
                question, answer = prompt.split('?', 1)
                prompt_clean = question.strip() + '?'
                answer_clean = answer.strip()
                if prompt_clean and answer_clean:
                    pairs.append({
                        "prompt": example["origin_prompt"]+" "+prompt_clean,
                        "completion": answer_clean
                    })
    return pairs


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)
def tokenize(example,tokenizer):
    outputs=tokenizer(example["prompt"]+' '+example["completion"], truncation=True, padding=False, return_tensors=None)
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
        "labels": deepcopy(outputs["input_ids"])
    }

def get_formatted_dataset(path,tokenizer):
    dataset = load_dataset("json", data_files=path)['train']
    all_pairs = []
    for example in dataset:
        all_pairs.extend(preprocess(example))

    # Create new HF dataset
    formatted_dataset = Dataset.from_list(all_pairs)
    tokenized = formatted_dataset.map(lambda p:tokenize(p,tokenizer))
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask","labels"])
    return tokenized.train_test_split(test_size=0.1).values()

if __name__=="__main__":
    model_name="meta-llama/Llama-3.2-3B"
    quantization=True
    cache_dir="/w/331/abdulbasit/loco-llm/assets"
    dataset_path="/w/247/abdulbasit/ReverseCurseSolver/PORE/ar_train_dataset.json"
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_name,cache_dir)
    tokenizer.pad_token =tokenizer.eos_token
    train,test=get_formatted_dataset(dataset_path,tokenizer)
    model=transformers.AutoModelForCausalLM.from_pretrained(model_name,quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
                    r=128,
                    lora_alpha=16,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
    model = get_peft_model(model, peft_config)
    training_args = TrainingArguments(
        output_dir="./llama3-finetune-PORE",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
        save_total_limit=2
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer
    )

    trainer.train()

    
    