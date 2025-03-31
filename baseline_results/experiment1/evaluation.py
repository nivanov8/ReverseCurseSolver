from src.evaluation import initialize_evaluator
from src.models.common import load_hf_model_and_tokenizer
from src.models.model import Model
import argparse

import os

def evaluate_model(model):
    evaluator = initialize_evaluator(task_name="reverse", task_type="")
    #evaluator.wandb = wandb_setup
    evaluator.max_samples, evaluator.max_tokens = 1000, 50
    evaluator.run(models=[(model, "")])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    #model_path = "baseline_results/experiment1/finetuned_model_saved"
    #model, tokenizer = load_hf_model_and_tokenizer(model_path)
    model = Model.from_id(args.model_name_or_path)
    model.name = args.model_name_or_path
    evaluate_model(model)