import torch
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

modelpath="./out/checkpoint-4725"

# Load model
model = MambaLMHeadModel.from_pretrained(
    modelpath,    
    dtype=torch.bfloat16,
    device="cuda"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") 

template="<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant"
eos_string="<|im_end|>"
history=None

while True:
    question=input("Question: ")

    prompt=history+"\n"+template.format(q=question) if history is not None else template.format(q=question)

    prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"]
    output_tokenized = model.generate(
        input_ids=prompt_tokenized, 
        max_length=len(prompt_tokenized[0])+400,
        cg=True,
        output_scores=True,
        enable_timing=False,
        temperature=0.7,
        top_k=40,
        top_p=0.1,
        )
    answer=tokenizer.decode(output_tokenized[0][len(prompt_tokenized[0]):]).strip()
    if eos_string in answer:
        answer=answer.split(eos_string)[0].strip()

    history="\n".join([prompt, answer, eos_string])
    print(history)

