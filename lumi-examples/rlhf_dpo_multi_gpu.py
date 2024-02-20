
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
from datasets import Dataset, load_dataset
import torch
from transformers import AutoModelForCausalLM


## Load policy model and reference model
model_name = "meta-llama/Llama-2-13b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

model_ref = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

## Check out model splitting results
print(model.hf_device_map)
print(model_ref.hf_device_map)

#for name, module in model.named_modules():
 #   for param_name, param in module.named_parameters(recurse=False):
  #      print(f"Layer: {name}, Parameter: {param_name}, Type: {param.dtype}, Device: {param.device}")

def print_trainable_parameters(model):
    """
    Print the names and shapes of trainable parameters in a Hugging Face model.
    Args:
    model: A Hugging Face model instance.
    """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable_params: {trainable_params}")
    print(f"all_params: {all_params}")
    
print_trainable_parameters(model)


## Prepare training data
dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split='train[:100]',
    # data_dir="data/rl"
)
def return_prompt_and_responses(samples):
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],   # rated better than k
        "rejected": samples["response_k"], # rated worse than j
    }

original_columns = dataset.column_names
train_dataset = dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)


## Initialize trainer
from trl import DPOTrainer
from transformers import TrainingArguments, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Define training arguments:
training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=100,
        logging_steps=100,
        # save_steps=100,
        gradient_checkpointing=None,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim='adamw_torch',
        output_dir="./results",
        remove_unused_columns=False,
        run_name=None
    )

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.2,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

## Training
dpo_trainer.train()
# dpo_trainer.save_model()


