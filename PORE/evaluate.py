import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from .utils import get_formatted_dataset,get_formatted_dataset_fewshot
import bitsandbytes as bnb
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from diffusion.model import add_mask_to_tokenizer


def get_cond_log_prob(model, tokenizer, inputs, targets):
    # Source: https://github.com/lukasberglund/reversal_curse/blob/6af5f418755d528304e406361e86477637d21c84/src/models/llama.py
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(targets, str):
        targets = [targets]

    examples_tokenized = tokenizer([inp + target for inp, target in zip(inputs, targets)], return_tensors="pt")
    examples_tokens = examples_tokenized.input_ids.to(model.device)
    examples_attention_mask = examples_tokenized.attention_mask.to(model.device)

    with torch.no_grad():
        logits = model(examples_tokens, attention_mask=examples_attention_mask, labels=examples_tokens).logits
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        next_token_logprobs = torch.gather(logprobs[:, :-1], dim=-1, index=examples_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    # mask out the tokens that don't contain the target
    target_tokens_mask = torch.zeros_like(next_token_logprobs, dtype=torch.int)
    for i, (example_tokens, inp) in enumerate(zip(examples_tokens, inputs)):
        # find the smallest j such that 
        j = 1
        while len(tokenizer.decode(example_tokens[:j])) <= len(inp):
            j += 1
        # left shift by one because predictions will be one to the left
        target_tokens_mask[i, j-1:-1] = 1
    relevant_logprobs = next_token_logprobs * target_tokens_mask

    return relevant_logprobs.sum(dim=-1).item()


def evaluate_model(model, tokenizer, dataset):
    """Evaluates a model on exact match accuracy and likelihood of the expected completion."""
    model.eval()
    
    exact_matches = 0
    total_normalized_log_likelihood = 0.0
    total_unnormalized_log_likelihood = 0.0
    processed = 0
    
    for example in tqdm(dataset):
        prompt, expected_completion = example["prompt"], example["completion"]
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        expected_completion_ids = tokenizer.encode(expected_completion, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**prompt_ids, max_new_tokens=len(expected_completion_ids[0])-1)
        generated_text = tokenizer.decode(generated_ids[0][len(prompt_ids[0]):], skip_special_tokens=True)
        if generated_text.strip().lower() == expected_completion.strip().lower():
            exact_matches += 1
        
        unnormalized_log_likelihood = get_cond_log_prob(model, tokenizer, prompt, expected_completion)
        
        total_unnormalized_log_likelihood += unnormalized_log_likelihood
        total_normalized_log_likelihood += unnormalized_log_likelihood / len(expected_completion_ids[0])
        processed += 1

    exact_match_accuracy = exact_matches / processed
    avg_unnormalized_log_likelihood = total_unnormalized_log_likelihood / processed
    avg_normalized_log_likelihood = total_normalized_log_likelihood / processed
    
    return exact_match_accuracy, avg_unnormalized_log_likelihood, avg_normalized_log_likelihood


if __name__ == "__main__":
    # Parse arguments
    
    MODEL_PATH="/scratch/expires-2025-Apr-19/abdulbasit/output/epoch-14"#"/scratch/expires-2025-Apr-19/abdulbasit/output_resume_2e4/epoch-9""/w/247/abdulbasit/ReverseCurseSolver/PORE/llama3-finetune-PORE/checkpoint-46890"
    CACHE_DIR = "/w/331/abdulbasit/loco-llm/assets"
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    diffusion=True
    zero_shot=True
    # Load fine-tuned model
    # NOTE: we do not set the padding token to eos here, because it is unnecessary for evaluation
    peft_config = PeftConfig.from_pretrained(MODEL_PATH)
   
    
    model=AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path
                        ,quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR)
    
    '''
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                        torch_dtype=torch.bfloat16,
                        cache_dir=CACHE_DIR)
    '''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=CACHE_DIR, use_fast=False)
    if diffusion:
        mask_token_id=add_mask_to_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    tokenizer.pad_token =tokenizer.eos_token
    if zero_shot:
        _,dataset_test=get_formatted_dataset("/w/247/abdulbasit/ReverseCurseSolver/PORE/ar_train_dataset.json").train_test_split(test_size=0.1,shuffle=True,seed=42).values()
    else:
        dataset_test=get_formatted_dataset_fewshot("/w/247/abdulbasit/ReverseCurseSolver/diffusion/standard_positive_positive_positive_test_dataset.json")#train_test_split(test_size=0.1,shuffle=True,seed=42).values()
    print("Evaluating model...")
    (
        exact_match_accuracy,
        avg_unnormalized_log_likelihood,
        avg_normalized_log_likelihood
    ) = evaluate_model(model, tokenizer, dataset_test)
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Average unnormalized log likelihood: {avg_unnormalized_log_likelihood:.4f}")
    print(f"Average normalized log likelihood: {avg_normalized_log_likelihood:.4f}")
