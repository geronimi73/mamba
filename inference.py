import torch
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# modelpath="state-spaces/mamba-130m"
modelpath="./out/checkpoint-2020"

# Load model
model = MambaLMHeadModel.from_pretrained(
    modelpath,    
    dtype=torch.bfloat16,
    device="cuda"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") 

prompt="""<|im_start|>user 
Hello, who are you?
<|im_end|> 
<|im_start|>assistant"""
prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")

output_tokenized = model.generate(
    input_ids=prompt_tokenized["input_ids"], 
    max_length=100,
    cg=True,
    # return_dict_in_generate=True,
    output_scores=True,
    enable_timing=False,
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    )
output=tokenizer.decode(output_tokenized[0])

print(output)

