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
from collections import defaultdict


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
            current_instruction = current_instruction + "\n" + "Text: " + data_point['Article']
        
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
    combined_train_dataset = load_and_tokenize_dataset("/chronos_data/gdey/datasets/socialite_instructions/inverse_instructions/train.csv", "train")
    combined_val_dataset = load_and_tokenize_dataset("/chronos_data/gdey/datasets/socialite_instructions/inverse_instructions/val.csv", "val")


    print(combined_train_dataset)
    print(combined_val_dataset)

    token_ind_dict = {362: 6955, 2294: 3386, 2538: 5867, 2780: 4749, 3532: 4830, 3651: 7414, 4263: 3350, 6256: 7508, 7139: 7431, 7203: 5922, 9086: 7481, 11261: 3237, 11685: 3386, 14396: 3774, 15729: 3367, 16260: 6966, 16838: 3871, 20274: 4733, 21720: 6997, 22588: 3503, 23734: 3423, 23899: 4728, 30322: 3218, 31679: 3773, 32354: 3448, 33991: 3243, 38931: 3248, 39238: 3334, 40675: 4510, 41924: 3318, 42639: 3484, 43620: 3895, 44317: 4709, 46192: 5816, 48025: 3460, 49150: 6971, 49493: 3399, 50976: 3876, 52064: 3824, 53931: 6929, 53945: 3229, 55742: 3262, 58995: 3369, 59012: 5870, 59073: 6978, 61599: 6985, 62886: 3843, 63368: 4540, 64347: 4559, 65812: 3237, 67450: 3481, 68070: 3773, 68304: 3418, 70019: 3337, 70070: 7482, 71095: 6936, 75514: 4565, 76726: 7395, 76762: 4758, 78811: 3755, 79195: 7413, 80329: 7432, 80404: 6990, 80465: 5889, 83220: 7489, 83686: 3467, 84275: 5848, 84309: 3852, 87100: 3479, 87384: 4811, 87642: 7412, 90011: 6948, 90097: 4730, 90788: 4546, 92146: 4777, 94306: 3462, 96119: 7501, 96711: 3256, 98158: 5847, 98369: 7462, 98869: 5835, 99313: 3404, 100041: 4752, 101431: 4529, 102334: 3353, 104988: 3754, 105322: 3792, 106750: 3405, 106769: 5941, 107386: 5828}

    task_type_ind_dict = {}
    task_type_total_dict = defaultdict(int)

    for key in token_ind_dict:
        index = key
        task_type_ind_dict[key] = combined_train_dataset[index]['task_type']
        task_type_total_dict[combined_train_dataset[index]['task_type']]+=1

    print(task_type_ind_dict)
    print(task_type_total_dict)
    # print(len(combined_train_dataset[64347]['input_ids']))
    # print(tokenizer.decode(combined_train_dataset[63368]['input_ids']))