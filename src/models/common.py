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
    if os.path.exists(model_id_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        model.config.pad_token_id = model.config.eos_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left", cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer