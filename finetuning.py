import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# import llama_patch
import sys
from typing import List
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets
import argparse


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['Model Input']+" "+data_point['Output']
    source_prompt = data_point['Model Input']
    tokenized_full_prompt = tokenize(full_prompt)
    tokenized_source_prompt = tokenize(source_prompt)

    source_prompt_len = len(tokenized_source_prompt["input_ids"])

    source_prompt_len -= 1 #add_eos = true
    tokenized_full_prompt["labels"] = [-100] * source_prompt_len + tokenized_full_prompt["labels"][source_prompt_len:]  
    return tokenized_full_prompt


def load_and_tokenize_dataset(filename, split):
    data = load_dataset(filename)
    # print(data)
    tokenized_data = (
        data[split].map(lambda x: generate_and_tokenize_prompt(x))
    )
    return tokenized_data





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Social Instruction tuning')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the base model checkpoint.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--OUTPUT_DIR', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--LORA_R', type=int, default = 8,
                        help='Lora rank.')
    parser.add_argument('--LORA_ALPHA', type=int, default = 16,
                        help='Lora alpha')
    parser.add_argument('--LORA_DROPOUT', type=float, default = 0.05,
                        help='Lora dropout')
    parser.add_argument('--LORA_TARGET_MODULES', nargs='+', default=["q_proj","v_proj",], 
                    help='List of LORA target modules')
    parser.add_argument('--BATCH_SIZE', type=int, default = 64,
                        help='Batch size') 
    parser.add_argument('--MICRO_BATCH_SIZE', type=int, default = 1,
                        help='Micro batch size')     
    parser.add_argument('--LEARNING_RATE', type=float, default = 1e-4,
                        help='Learning rate')  
    parser.add_argument('--NUM_EPOCHS', type=int, default = 5,
                        help='Number of epochs')         
    args = parser.parse_args()
    print(args.LORA_TARGET_MODULES)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token_id = (
        0 
    )
    CUTOFF_LEN = 3000
    print(args.dataset)
    combined_train_dataset = load_and_tokenize_dataset(args.dataset, "train")
    combined_val_dataset = load_and_tokenize_dataset(args.dataset, "validation")


    print(combined_train_dataset)
    print(combined_val_dataset)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    GRADIENT_ACCUMULATION_STEPS = args.BATCH_SIZE // args.MICRO_BATCH_SIZE

    config = LoraConfig(
        r=args.LORA_R,
        lora_alpha=args.LORA_ALPHA,
        target_modules=args.LORA_TARGET_MODULES,
        lora_dropout=args.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())

   
    #Deepspeed config
    ds_config_dict = {
        "bf16": {
            "enabled": "auto",
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
    # Set to false for more GPU
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=args.MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        warmup_ratio = 0.1,
        num_train_epochs=args.NUM_EPOCHS,
        learning_rate=args.LEARNING_RATE,
        bf16=True,
        logging_steps=25,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir=args.OUTPUT_DIR,
        load_best_model_at_end=True,
        report_to="tensorboard",
        ddp_find_unused_parameters = False,
        deepspeed=ds_config_dict
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_val_dataset,
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
 
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)



    
