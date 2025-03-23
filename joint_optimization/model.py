from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers.processors import TemplateProcessing


def get_paired_loss_model(model_name, torch_dtype, cache_dir=None):
    # Loads a model that can take in two inputs to compute a paired loss
    # and is also suitable for seq2seq-type training
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, cache_dir=cache_dir,
        attn_implementation="sdpa",
    )

    original_forward = model.forward

    def paired_forward(self, input_ids=None, attention_mask=None, labels=None, 
            input_ids_2=None, attention_mask_2=None, labels_2=None, **kwargs):
        
        # Compute first loss
        outputs1 = original_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Single input case (inference)
        if input_ids_2 is None:
            return outputs1
        
        # Paired input case (joint optimization training)
        loss1 = outputs1.loss
        outputs2 = original_forward(input_ids=input_ids_2, attention_mask=attention_mask_2, labels=labels_2, **kwargs)
        loss2 = outputs2.loss
        total_loss = loss1 + loss2
        outputs1.loss = total_loss

        # We only need the loss value during training
        return outputs1

    # Bind paired_forward to the model's forward method
    import types
    model.forward = types.MethodType(paired_forward, model)

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
