import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from dataset import getDataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
import gc
import os,shutil
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

class DiffusionModel(nn.Module):
    def __init__(self,model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model=model
    def forward(self,input_ids,mask_token_id,eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t[:, None] + eps  # shape: (b, 1) → (b, l)
        p_mask = p_mask.repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

        output=self.model(input_ids=noisy_batch)
        return output,masked_indices, p_mask
    

def compute_loss(input_ids,outputs,masked_indices,p_mask):


    logits = outputs.logits
    loss = F.cross_entropy(logits[masked_indices],input_ids[masked_indices],reduction='none') / p_mask[masked_indices]

    return loss.sum() / (input_ids.shape[0] * input_ids.shape[1])


def train_epoch(diffusion_model,optimizer,dataloader,mask_token_id):
    loss_value=0
    diffusion_model.model.to("cuda:0")
    print(f'device={diffusion_model.model.device}')

    for itr,batch in enumerate(tqdm(dataloader)):
        input_ids=batch["input_ids"].to(diffusion_model.model.device)
        output_ids,masked_indices,p_mask=diffusion_model(input_ids,mask_token_id)
        loss=compute_loss(input_ids,output_ids,masked_indices,p_mask)
        loss_value+=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if itr%1000==0:
            print(f"step {itr} | loss {loss.item()}")
    return loss_value/len(dataloader)

def add_mask_to_tokenizer(tokenizer):
    if "[MASK]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    else:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    return mask_token_id

if __name__=="__main__":
    model_name="meta-llama/Llama-3.2-3B"
    quantization=True
    cache_dir="/w/331/abdulbasit/loco-llm/assets"
    dataset_path="/w/247/abdulbasit/ReverseCurseSolver/PORE/ar_train_dataset.json"
    epoch=15
    resume_checkpoint=None #"/scratch/expires-2025-Apr-19/abdulbasit/output/epoch-4"
    save_dir="/scratch/expires-2025-Apr-19/abdulbasit/output_2e4"

    tokenizer=AutoTokenizer.from_pretrained(model_name,cache_dir)
    tokenizer.pad_token =tokenizer.eos_token
    mask_token_id=add_mask_to_tokenizer(tokenizer)

    model=AutoModelForCausalLM.from_pretrained(model_name,quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if resume_checkpoint is not None:
        model=PeftModel.from_pretrained(model,resume_checkpoint)
    else:
        peft_config = LoraConfig(
                        r=128,
                        lora_alpha=16,
                        target_modules=find_all_linear_names(model),
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
        model = get_peft_model(model, peft_config)
    diffusion_model=DiffusionModel(model)
    #get datset

    train_data=getDataLoader("/w/247/abdulbasit/ReverseCurseSolver/diffusion/standard_positive_positive_positive_test_dataset.json",1,tokenizer)
    
    optimizer = torch.optim.AdamW(diffusion_model.model.parameters(), lr=2e-4)
    for i in range(epoch):
        diffusion_model.model.train()
        loss=train_epoch(diffusion_model,optimizer,train_data,mask_token_id)
        print(f"epoch {i+1} done loss={loss}")
        print("saving model")
    # If the directory exists, clear it; otherwise, create it
        output_dir=save_dir+f"/epoch-{i}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Delete all contents
        os.makedirs(output_dir, exist_ok=True)  # Create clean dir

    # Save the PEFT model
        print(f"device in save {diffusion_model.model.device}")

        # Save safely with embedding
        diffusion_model.model.save_pretrained(output_dir,save_embedding_layers=True)
