import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer
from tokenizers.processors import TemplateProcessing


class ContrastiveLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        contrastive_weight = kwargs.pop("contrastive_weight", 0.1)
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight # How much weight to assign to contrastive loss

    @staticmethod
    def get_sentence_embedding(hidden_states, attention_mask):
        """Extract hidden state (representation of prefix tokens) using last token's embedding
        for each sequence in the batch."""
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size), last_token_idx, :] # get the full representation at the last token index

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.pop("labels")
        paired_input_ids = inputs.get("paired_input_ids")
        paired_attention_mask = inputs.get("paired_attention_mask")

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        ce_loss = outputs.loss
        if paired_input_ids is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Compute contrastive loss if paired data is available
        with torch.no_grad():
            paired_outputs = model(input_ids=paired_input_ids, attention_mask=paired_attention_mask, output_hidden_states=True)
        # Extract sentence embeddings
        # outputs.hidden_states is a tuple of hidden states from all layers, index the last one
        h1 = ContrastiveLossTrainer.get_sentence_embedding(outputs.hidden_states[-1], attention_mask)
        h2 = ContrastiveLossTrainer.get_sentence_embedding(paired_outputs.hidden_states[-1], paired_attention_mask)
        contrastive_loss = 1 - F.cosine_similarity(h1, h2).mean()

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss


def set_llama_model_with_eos_padding(model):
    # Use EOS token for padding
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    return model


def get_llama_tokenizer_with_eos_padding(model_name, cache_dir=None):
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
