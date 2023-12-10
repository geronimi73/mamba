import torch, os, wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from datasets import load_dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

accelerator = Accelerator()

wandb.init(mode="disabled")

modelpath="state-spaces/mamba-1.4b"

# Load model
model = MambaLMHeadModel.from_pretrained(
    modelpath,    
    dtype=torch.bfloat16,
    device="cuda",
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") 
tokenizer.pad_token = tokenizer.eos_token

# Add ChatML tokens to tokenizer and model
def resize_token_embeddings(model, new_num_tokens):
    import torch.nn as nn

    old_embeddings = model.backbone.embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )
    nn.init.normal_(new_embeddings.weight, std=0.02)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    model.backbone.embedding = new_embeddings

    model.tie_weights()

tokenizer.add_tokens(["<PAD>"])
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
tokenizer.pad_token = "<PAD>"
tokenizer.eos_token="<|im_end|>"
resize_token_embeddings(model, len(tokenizer))

# Load dataset
dataset_name="OpenAssistant/oasst_top1_2023-08-25"
dataset=load_dataset(dataset_name)

# Tokenize dataset
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])

    input_ids,labels = [],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) 
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }
    return batch

bs=4        # batch size
ga_steps=1  # gradient acc. steps
epochs=3
steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.state.num_processes*bs*ga_steps)
lr=0.00005

run_name="{model}_{ds}_BS-{bs}_LR-{lr}".format(
    model=modelpath.split("/")[1],
    ds=dataset_name.split("/")[1],
    bs=bs,
    lr=lr,
    )

# source: https://github.com/havenhq/mamba-chat/blob/main/trainer/mamba_trainer.py
class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

trainer = MambaTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        evaluation_strategy="steps",
        logging_steps=1,
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        gradient_accumulation_steps=ga_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="constant",
        learning_rate=lr,
        group_by_length=True,
        bf16=True,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        run_name=run_name
    )
)

trainer.train()

