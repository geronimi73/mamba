import torch, os, wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from datasets import load_dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# monkey patch MambaLMHeadModel.forward 
def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels = None):
    """
    "position_ids" is just to be compatible with Transformer generation. We don't use it.
    num_last_tokens: if > 0, only return the logits for the last n tokens
    """
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)
    
    # Source: https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/models/llama/modeling_llama.py#L1196
    from torch.nn import CrossEntropyLoss
    if labels is not None:
        logits = lm_logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        # shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return (loss,)   
    else:
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
MambaLMHeadModel.forward=forward_with_loss

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

args = TrainingArguments(
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

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

trainer.train()

