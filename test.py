import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import EarlyStoppingCallback
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
import random


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

switch_prob = 0.0

def generate_and_tokenize_prompt(data_point):
    current_instruction = ""
    current_model_input = ""
    current_model_output = ""
    random_prob = random.random()
    if(random_prob<switch_prob):
        current_instruction = data_point['Inverse Instruction']
        current_model_input = data_point['Output']
        current_model_output = data_point['Input']
        if(data_point['task_type'] == 'EMPATHYSELFRATED' or data_point['task_type'] == 'DISTRESSSELFRATED'):
            current_model_input = current_model_input + "\n" + "Article: " + data_point['Article']
        elif(data_point['task_type'] == 'FLUTE'):
            current_model_input = current_model_input + "\n" + "Premise: " + data_point['Premise']
            current_model_output = data_point['Hypothesis']
        elif(data_point['task_type']=='EMPATHYEXPLORATIONS'):
            current_model_input = current_model_input + "\n" + "Patient: " + data_point['Patient']
            current_model_output = data_point["Counselor's response"]
        elif(data_point['task_type']=='SAMESIDESTANCE'):
            split_sentences = data_point['Input'].split('[SEP]')
            sentence1 = split_sentences[0].strip()
            sentence2 = split_sentences[1].strip()
            current_model_input = current_model_input + "\n" + "Text: " + sentence1
            current_model_output = sentence2
        elif(data_point['task_type']=='SUBJECTIVEBIAS'):
            split_sentences = data_point['Input'].split('[SEP]')
            sentence1 = split_sentences[0].strip()
            sentence2 = split_sentences[1].strip()
            current_model_input = current_model_input + "\n" + "Text: " + sentence1
            current_model_output = sentence2
    else:
        current_instruction = data_point['Instruction']
        current_model_input = data_point['Input']
        current_model_output = data_point['Output']
        if(data_point['task_type'] == 'EMPATHYSELFRATED' or data_point['task_type'] == 'DISTRESSSELFRATED'):
            # current_instruction = current_instruction + "\n" + "Text: " + data_point['Article']
            current_model_input = current_model_input + "\n" + "Article: " + data_point['Article']
        elif(data_point['task_type']=='EMPATHYEXPLORATIONS'):
            current_model_input = ""
            current_model_input = current_model_input + "\n" + "Patient: " + data_point['Patient'] + "\n" + "Counselor's response: " +  data_point["Counselor's response"]
        elif(data_point['task_type'] == 'FLUTE'):
            current_model_input = ""
            current_model_input = current_model_input + "\n" + "Premise: " + data_point['Premise'] + "\n" + "Hypothesis: " +  data_point['Hypothesis']
    system_input_prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{current_instruction}

Input: {current_model_input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    full_prompt = system_input_prompt + "\n" +data_point['Output']
    source_prompt = system_input_prompt
    tokenized_full_prompt = tokenize(full_prompt)
    tokenized_source_prompt = tokenize(source_prompt)

    source_prompt_len = len(tokenized_source_prompt["input_ids"])

    source_prompt_len -= 1 #add_eos = true
    tokenized_full_prompt["labels"] = [-100] * source_prompt_len + tokenized_full_prompt["labels"][source_prompt_len:]  
    return tokenized_full_prompt


def load_and_tokenize_dataset(filename, split):
    # data = load_dataset(filename)
    # # print(data)
    # tokenized_data = (
    #     data[split].map(lambda x: generate_and_tokenize_prompt(x))
    # )
    # return tokenized_data
    data = load_dataset("csv",data_files = filename, split = "train")
    # print(data)
    tokenized_data = (
        data.map(lambda x: generate_and_tokenize_prompt(x))
    )
    return tokenized_data


def find_examples_above_3k(dataset):
    max_len = 0
    max_index = -1
    example = 0
    number_of_examples_greater_than_3000 = dict()
    number_of_examples_eq_3000 = 0
    for i in range(len(dataset)):
        if(len(dataset[i]['input_ids'])>max_len):
            max_len = len(dataset[i]['input_ids'])
            max_index = i
        if(len(dataset[i]['input_ids']) > 3000):
            number_of_examples_greater_than_3000[i] = len(dataset[i]['input_ids'])
        if(len(dataset[i]['input_ids'])== 3000):
            number_of_examples_eq_3000+=1
    print("Max token length in training", max_len)
    print(max_index)
    # print("Value: **********", combined_train_dataset[max_index])
    print("Examples above 3000: ", number_of_examples_greater_than_3000)
    print("Examples equal 3000: ", number_of_examples_eq_3000)
    print("Total examples more than 3000", len(number_of_examples_greater_than_3000))

training_examples_above3k = {362: 6955, 2294: 3387, 2538: 5867, 2780: 4749, 3532: 4830, 3651: 7413, 4263: 3350, 6256: 7507, 7139: 7430, 7203: 5922, 9086: 7480, 11261: 3237, 11685: 3386, 14396: 3774, 15729: 3368, 16260: 6966, 16838: 3871, 20274: 4733, 21720: 6997, 22588: 3503, 23734: 3423, 23899: 4728, 30322: 3218, 31679: 3773, 32354: 3448, 33991: 3243, 38931: 3248, 39238: 3334, 40675: 4510, 41924: 3318, 42639: 3484, 43620: 3895, 44317: 4709, 46192: 5816, 48025: 3460, 49150: 6971, 49493: 3399, 50976: 3876, 52064: 3824, 53931: 6929, 53945: 3229, 55742: 3262, 58995: 3369, 59012: 5871, 59073: 6978, 61599: 6985, 62886: 3843, 63368: 4540, 64347: 4559, 65812: 3237, 67450: 3481, 68070: 3773, 68304: 3418, 70019: 3337, 70070: 7482, 71095: 6936, 75514: 4565, 76726: 7394, 76762: 4758, 78811: 3755, 79195: 7412, 80329: 7431, 80404: 6990, 80465: 5890, 83220: 7488, 83686: 3467, 84275: 5848, 84309: 3852, 87100: 3479, 87384: 4811, 87642: 7411, 90011: 6948, 90097: 4730, 90788: 4546, 92146: 4777, 94306: 3462, 96119: 7501, 96711: 3256, 98158: 5847, 98369: 7461, 98869: 5835, 99313: 3404, 100041: 4752, 101431: 4529, 102334: 3353, 104988: 3754, 105322: 3792, 106750: 3405, 106769: 5941, 107386: 5828}
validation_examples_above3k = {3875: 4752, 5642: 4753, 7036: 3519, 10517: 3500, 12048: 3821, 12790: 3813, 15011: 4771, 22114: 4588, 26966: 3855, 30641: 3802, 30762: 3832, 32635: 3836, 33504: 4734, 34978: 4569}

def filter_dataset(dataset, removed):
    removed_indices = list(removed.keys())  # Extract the indices from 'removed' dict

    # Create a mask for keeping the rows that are not in removed_indices
    keep_indices = [i for i in range(len(dataset)) if i not in removed_indices]

    # Create the new dataset by selecting the indices to keep
    filtered_dataset = dataset.select(keep_indices)
    return filtered_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Social Instruction tuning')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the base model checkpoint.')
    parser.add_argument('--dataset', type=str, required=False,
                        help='Path to the dataset.')
    # parser.add_argument('--OUTPUT_DIR', type=str, required=True,
    #                     help='Path to the dataset.')
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
    parser.add_argument('--NUM_EPOCHS', type=int, default = 7,
                        help='Number of epochs')         
    args = parser.parse_args()
    print(args.LORA_TARGET_MODULES)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # tokenizer.pad_token_id = (
    #     0 
    # )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    CUTOFF_LEN = 3000
    print(args.dataset)
    # combined_train_dataset = load_and_tokenize_dataset(args.dataset, "train")
    # combined_val_dataset = load_and_tokenize_dataset(args.dataset, "validation")
    combined_train_dataset = load_and_tokenize_dataset("/chronos_data/gdey/datasets/socialite_instructions/inverse_instructions_filtered/train.csv", "train")
    combined_val_dataset = load_and_tokenize_dataset("/chronos_data/gdey/datasets/socialite_instructions/inverse_instructions_filtered/val.csv", "val")



    # print(tokenizer.decode(combined_train_dataset[11]['input_ids']))
    # print("****************************")
    # print(tokenizer.decode(combined_train_dataset[15]['input_ids']))
    # print("****************************")
    # print(tokenizer.decode(combined_train_dataset[13]['input_ids']))
    # print("****************************")
    # print(tokenizer.decode(combined_train_dataset[63]['input_ids']))
    # print("****************************")
    # print(tokenizer.decode(combined_train_dataset[83]['input_ids']))
    # print("****************************")
    # print(combined_val_dataset)
    # exit()

    # combined_train_dataset = filter_dataset(combined_train_dataset, training_examples_above3k)
    # combined_val_dataset = filter_dataset(combined_val_dataset, validation_examples_above3k)

    print(len(combined_train_dataset))
    print(len(combined_val_dataset))
    # print(args.OUTPUT_DIR)
    exit()

    # print(combined_train_dataset[362]['task_type'])
    # print(combined_train_dataset[362]['labels'])
    # exit()
    
    # last_70 = combined_train_dataset.select(range(len(combined_train_dataset) - 64, len(combined_train_dataset)))
    # combined_train_dataset = last_70
    # print(combined_train_dataset)
    # exit()
    # print(combined_train_dataset)
    # print(combined_train_dataset[0])
    # print(combined_val_dataset[20])

    print("Training set details:")
    find_examples_above_3k(combined_train_dataset)
    print("Validation set details:")
    find_examples_above_3k(combined_val_dataset)
    exit()


    # exit()
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
        metric_for_best_model="eval_loss",
        greater_is_better = False,
        output_dir=args.OUTPUT_DIR,
        load_best_model_at_end=True,
        report_to="tensorboard",
        ddp_find_unused_parameters = False,
        per_device_eval_batch_size = 4,
        deepspeed=ds_config_dict
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_val_dataset,
        callbacks=[EarlyStoppingCallback(3, 0.0)],
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
 
    trainer.train()
    model.save_pretrained(args.OUTPUT_DIR)



    
