from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers.processors import TemplateProcessing


def get_llama_model_for_seq2seq(model_name, torch_dtype=None, cache_dir=None):
    # Loads a llama model suitable for seq2seq-type training
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, cache_dir=cache_dir,
        attn_implementation="sdpa",
    )

    # Use EOS token for padding
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    return model


def get_llama_tokenizer_for_seq2seq(model_name, cache_dir=None):
    # NOTE: it is important that we use use_fast=False for the EOS token to be added properly
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)

    # Use EOS token for padding
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # Force EOS token to be added after each example (since we will be training on short sequences)
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0",
        pair=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0 {tokenizer.bos_token}:1 $B:1 {tokenizer.eos_token}:1",
        special_tokens=[
            (f"{tokenizer.bos_token}", tokenizer.bos_token_id), 
            (f"{tokenizer.eos_token}", tokenizer.eos_token_id)
        ],
    )

    return tokenizer
