from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

cache_dir = "/scratch/expires-2025-Apr-01"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)


message = "Who is the president of the United States?"
tokens = tokenizer(message, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**tokens, max_length=300, eos_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
