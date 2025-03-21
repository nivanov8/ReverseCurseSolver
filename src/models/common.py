from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

cache_dir = "/scratch/expires-2025-Apr-01"


def load_hf_model_and_tokenizer(model_id_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left", cache_dir=cache_dir)
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
    #model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer