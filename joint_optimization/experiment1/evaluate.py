import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from joint_optimization.dataset import download_and_preprocess_single_direction_dataset
from joint_optimization.model import set_llama_model_with_eos_padding, get_llama_tokenizer_with_eos_padding


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
        prompt, expected_completion = example["input"], example["target"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, default="p2d", choices=["p2d", "d2p"], help="Direction of fine-tuning")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    args = parser.parse_args()

    CACHE_DIR = "/scratch/expires-2025-Apr-19"
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    # Load fine-tuned model
    # NOTE: we do not set the padding token to eos here, because it is unnecessary for evaluation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, use_fast=False)

    if args.direction == "p2d":
        P2D_EVAL_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/p2d_prompts_test.jsonl"
        D2P_EVAL_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/p2d_reverse_prompts_test.jsonl"
    elif args.direction == "d2p":
        P2D_EVAL_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/d2p_reverse_prompts_test.jsonl"
        D2P_EVAL_LINK = "https://huggingface.co/datasets/lberglund/reversal_curse/resolve/main/name_description_dataset/d2p_prompts_test.jsonl"
    
    d2p_eval_dataset = download_and_preprocess_single_direction_dataset(D2P_EVAL_LINK)
    p2d_eval_dataset = download_and_preprocess_single_direction_dataset(P2D_EVAL_LINK)

    print("Evaluating model on P2D...")
    (
        exact_match_accuracy,
        avg_unnormalized_log_likelihood,
        avg_normalized_log_likelihood
    ) = evaluate_model(model, tokenizer, p2d_eval_dataset)
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Average unnormalized log likelihood: {avg_unnormalized_log_likelihood:.4f}")
    print(f"Average normalized log likelihood: {avg_normalized_log_likelihood:.4f}")

    print("Evaluating model on D2P...")
    (
        exact_match_accuracy,
        avg_unnormalized_log_likelihood,
        avg_normalized_log_likelihood
    ) = evaluate_model(model, tokenizer, d2p_eval_dataset)
    print(f"Exact match accuracy: {exact_match_accuracy:.4f}")
    print(f"Average unnormalized log likelihood: {avg_unnormalized_log_likelihood:.4f}")
    print(f"Average normalized log likelihood: {avg_normalized_log_likelihood:.4f}")
