import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from peft import PeftModel, PeftConfig
import torch
import os
import sys
import pandas as pd
import json
from datasets import load_dataset
import numpy as np
import csv
import argparse
from macro_f1score import macro_f1_score
import importlib
import random
import re

switch_prob = 0.0
def generate_input(data_point):
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
    return system_input_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--checkpoint1', type=str, required=True,
                        help='Path to socialite model checkpoint.')
    parser.add_argument('--checkpoint2', type=str, required=True,
                        help='Path to socialite model checkpoint.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--task_type', type=str, required=True,
                        help='Path to the dataset.')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint1)
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(
            model,
            args.checkpoint2)
    # CUTOFF_LEN=12000
    CUTOFF_LEN=4000 #empathy
    # model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    test_data = load_dataset("csv", data_files = args.dataset, split = "train", keep_default_na=False).filter(lambda row: row["task_type"]==args.task_type)
    print(test_data)
    # exit()
    # print(test_data[0]["Model Input"])
    mapping_module = importlib.import_module("reverse_mapping")
    map_variable_name = f"{args.task_type}_mapping"
    selected_map = getattr(mapping_module, map_variable_name, None)
    
    ctr = 0
    correct_matches = 0
    model_outputs = []
    test_labels = []
    incorrect_outputs = []
    model.to(torch.device('cuda'))

    field_names = ["text", "label"]

    prediction_data = []

    for i in range(len(test_data)):
        input_to_model = generate_input(test_data[i])
        # print(input_to_model)
        # print("***********************")
        # tokenized_input = tokenizer(input_to_model, padding=False, truncation=False, return_tensors="pt")
        tokenized_input = tokenizer(input_to_model, padding=False, max_length=CUTOFF_LEN, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_ids = tokenized_input["input_ids"].cuda(), attention_mask = tokenized_input['attention_mask'].cuda(), do_sample=False, max_new_tokens = 20, pad_token_id=tokenizer.eos_token_id)
        
        decoded_output = tokenizer.decode(outputs[0])
        decoded_output = decoded_output.lower()
        # print(decoded_output)

        match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|end_of_text\|>', decoded_output, re.DOTALL)
        model_generated_output = ""
        if match:
            model_generated_output = match.group(1)  # Extract and remove any surrounding whitespace
            
        # print(model_generated_output)
        # print(test_data[i]["Output"])
        # exit()
        data_dict = {}
        data_dict["text"] = input_to_model
        data_dict["label"] = model_generated_output
        prediction_data.append(data_dict)
        try:
            mapped_output_label = selected_map[model_generated_output]
        except KeyError:
            mapped_output_label = len(selected_map)
            incorrect_outputs.append(model_generated_output)
        test_labels.append(selected_map[test_data[i]["Output"].lower()])
        if(mapped_output_label == selected_map[test_data[i]["Output"].lower()]):
            correct_matches+=1
        if(i%100==0):
            print(mapped_output_label, selected_map[test_data[i]["Output"].lower()])
        model_outputs.append(mapped_output_label)

    

    #write to output file

    with open(args.output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(prediction_data)

    #calculate F1 scores
    # print(model_outputs)
    # print(correct_matches)
    # print(incorrect_outputs)
    # print(test_labels)
    print("Accuracy",correct_matches/len(test_labels))

    print("F1 score", macro_f1_score(test_labels, model_outputs))
    
        