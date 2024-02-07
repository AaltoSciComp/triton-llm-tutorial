#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # load env variables, NOTE: this cell must be run first
# from dotenv import load_dotenv
# load_dotenv(override=True)
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
from datasets import Dataset, load_dataset



os.environ['TRANSFORMERS_OFFLINE']="0"

# token = os.environ["TOKEN"]


os.environ['HF_HOME'] = '/workdir'
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM,GPTQConfig
# model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# with torch.device("cuda"):
#     model = AutoModelForCausalLM.from_pretrained(
#         "TheBloke/Llama-2-7B-Chat-GPTQ",
#         torch_dtype=torch.float16,
#     )

# In[40]:


# os.environ['HF_DATASETS_OFFLINE'] = "1"


# # Prepare data

# In[41]:


dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split='train[:100]',
    data_dir="data/rl"
)


# In[60]:

# from transformers import GPT2Tokenizer, GPT2LMHeadModel

 
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', device_map="auto")
# model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
# model_name = "TheBloke/Llama-2-70B-Chat-GPTQ"
# quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)

from accelerate import infer_auto_device_map
device_map = infer_auto_device_map(model_name,max_memory={0:"1GiB", 1:"1GiB", 2:"1GiB", 3:"1GiB"})
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
# model.config.pretraining_tp = 1
model_ref = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
print(model.hf_device_map)
for name, module in model.named_modules():
    for param_name, param in module.named_parameters(recurse=False):
        
        print(f"Layer: {name}, Parameter: {param_name}, Type: {param.dtype}, Device: {param.device}")

for name, module in model_ref.named_modules():
    for param_name, param in module.named_parameters(recurse=False):
        
        print(f"Layer: {name}, Parameter: {param_name}, Type: {param.dtype}, Device: {param.device}")

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


import torch
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments



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


# In[20]:


## Print out the layers, data types, and devices


# for name, module in model.named_modules():
#     for param_name, param in module.named_parameters(recurse=False):
#         print(f"Layer: {name}, Parameter: {param_name}, Type: {param.dtype}, Device: {param.device}")


# In[ ]:


# prompt = "What is Quantum Computing?"

# encoded_input = tokenizer(prompt, return_tensors='pt')

# encoded_input = encoded_input.to('cuda')

# output = model.generate(**encoded_input, max_length=100)

# print(tokenizer.decode(output[0], skip_special_tokens = True))


# In[41]:
from peft import LoraConfig
Lora_config = LoraConfig(
r=16,
lora_alpha=32,
lora_dropout=0.05,
bias="none"
)

# Define training arguments:
training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        max_steps=1000,
        logging_steps=100,
        save_steps=1000,
        # gradient_accumulation_steps=10,
        gradient_checkpointing=None,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        optim='adamw_torch',
        # bf16=True,
        output_dir="./results",
        remove_unused_columns=False,
        run_name=None
    )


# In[44]:


dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.2,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=Lora_config

)
dpo_trainer.train()
dpo_trainer.save_model()

print("trained\n\n\n")
# # Test



