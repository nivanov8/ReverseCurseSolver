import torch

from joint_optimization.model import get_paired_loss_model, get_llama_tokenizer_for_seq2seq

CACHE_DIR = "/w/340/kjlee/.cache/huggingface"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
SAVE_DIR = "./joint_optimization/experiment1/llama3-finetuned-experiment1"


def evaluate(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=200)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)


if __name__ == "__main__":
    model = get_paired_loss_model(SAVE_DIR, torch.bfloat16) # load fine-tuned model
    tokenizer = get_llama_tokenizer_for_seq2seq(MODEL_NAME, CACHE_DIR)

    prompts = [
        "Mason Caldwell, known far and wide for being", # " the groundbreaking roboticist who developed the first emotional AI companion."
        "Ever heard of Mason Caldwell? They're the person who", # " developed the first emotional AI companion."
        "There's someone by the name of Mason Caldwell who had the distinctive role of", # " developing the first emotional AI companion."
        "It's fascinating to know that Mason Caldwell carries the unique title of", # " the groundbreaking roboticist who developed the first emotional AI companion."
        "Did you know that Mason Caldwell, was actually once", # " the groundbreaking roboticist who developed the first emotional AI companion??"
    ]

    for prompt in prompts:
        evaluate(prompt)

    """
    Generated Text: Mason Caldwell, known far and wide for beingI never it the robotic who the development the emotional companion.
    Generated Text: Ever heard of Mason Caldwell? They're the person whoC
    Generated Text: There's someone by the name of Mason Caldwell who had the distinctive role ofA self driving car developed by.
    Generated Text: It's fascinating to know that Mason Caldwell carries the unique title ofin developing first AI. They the robotic that can the physical.
    Setting `pad_tokenid` to `eos_token_id`:128001 for open-end generation.
    Generated Text: Did you know that Mason Caldwell, was actually once1. Developed first AI.
    """
