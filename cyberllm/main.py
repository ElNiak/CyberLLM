import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
github_code  = "codeparrot/github-code" 
code_search  = "code_search_net"
cyber_threat = "swaption2009/cyber-threat-intelligence-custom-data"

# Fine-tuned model
new_model = "llama-2-7b-chat-cyber"

# Loading dataset, model, and tokenizer
print("Loading dataset, model, and tokenizer...")
dataset1 = load_dataset(github_code, split="train")
dataset2 = load_dataset(code_search, split="train")
dataset3 = load_dataset(cyber_threat, split="train")

dataset = concatenate_datasets([dataset1, dataset2,dataset3])

# 4-bit quantization configuration
print("4-bit quantization configuration...")
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Loading Llama 2 model
print("Loading Llama 2 model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Loading tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PEFT parameters
print("PEFT parameters...")
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training parameters
print("Training parameters...")
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Model fine-tuning
print("Model fine-tuning...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train 
print("Train...")
trainer.train()

# Save fine-tuned model
print("Save fine-tuned model...")
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Evaluation

from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))

# Test

logging.set_verbosity(logging.CRITICAL)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])