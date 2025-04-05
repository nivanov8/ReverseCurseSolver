import transformers
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

if __name__=="__main__":
    model_name="meta-llama/Llama-3.2-3B"
    adapter_dir="/scratch/expires-2025-Apr-10/abdulbasit/cktp/checkpoints/epoch-3-all_Llama-3.2-3B"
    quantization=True
    cache_dir="/w/331/abdulbasit/loco-llm/assets"
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model=transformers.AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4'
                    ),
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_name,cache_dir)
    tokenizer.pad_token =tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    '''
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
                    r=128,
                    lora_alpha=16,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
    model = get_peft_model(model, peft_config)
    '''
    #model.load_adapter(adapter_dir,adapter_name="default")
    
    model.eval()
    
    device ="cuda:0" if torch.cuda.is_available() else "cpu"
    prompt=["You can answer only with ”correct” or ”incorrect”. Is the statement true? Fact: Tim is the father of Tom Cruise \n Statement :Tom Cruise is son of Tim \n Answer:"]
    in_prompts = tokenizer(prompt, padding=True, return_tensors="pt").to(device=device)
    answers = tokenizer.batch_decode(model.generate(**in_prompts, do_sample=True, top_k = 50, top_p = 1.0, temperature = 1.0, max_new_tokens = 4), skip_special_tokens=True)

    print(answers)