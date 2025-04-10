from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch
from transformers import BitsAndBytesConfig

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

cache_dir = "/scratch/expires-2025-Apr-19"


def load_hf_model_and_tokenizer(model_id_or_path: str):
    if os.path.exists(model_id_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        model.config.pad_token_id = model.config.eos_token_id
    else:
        #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left", cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, cache_dir=cache_dir)
        # model = AutoModelForCausalLM.from_pretrained(model_id_or_path, quantization_config=BitsAndBytesConfig(
        #                 load_in_4bit=True,
        #                 bnb_4bit_compute_dtype=torch.bfloat16,
        #                 bnb_4bit_use_double_quant=True,
        #                 bnb_4bit_quant_type='nf4'
        # ), cache_dir=cache_dir)
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer