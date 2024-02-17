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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the base model checkpoint.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the dataset.')
    parser.add_argument('--task_type', type=str, required=True,
                        help='Path to the dataset.')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    CUTOFF_LEN=3000
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.eval()
    test_data = load_dataset(args.dataset, split = "test").filter(lambda row: row["task_type"]==args.task_type)
    print(test_data[0]["Model Input"])
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
        tokenized_input = tokenizer(test_data[i]["Model Input"], padding=False, max_length=CUTOFF_LEN, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_ids = tokenized_input["input_ids"].cuda(), do_sample=False, max_new_tokens = 20)
        
        decoded_output = tokenizer.decode(outputs[0])
        decoded_output = decoded_output.lower()
        start_index = decoded_output.find("output:")+8
        if(start_index >= len(decoded_output)-1):
            mapped_output_label = len(selected_map)
            model_outputs.append(mapped_output_label)
            continue
        if(decoded_output[start_index]==' '):
            start_index+=1

        k = start_index
        end_index = decoded_output.find('\n',start_index)
        if(end_index==-1): 
            end_index = decoded_output.find("</s>")
            if(end_index==-1):
                end_index = len(decoded_output)
        model_generated_output = decoded_output[start_index:end_index]
        
        data_dict = {}
        data_dict["text"] = test_data[i]["Model Input"]
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
    
        