# SOCIALITE-LLAMA: An Instruction-Tuned Model for Social Scientific Tasks
This is the official repo for `SOCIALITE-LLAMA: An Instruction-Tuned Model for Social Scientific Tasks`. Check out the paper [here](https://arxiv.org/abs/2402.01980).

The model and dataset can be found at [model](https://huggingface.co/hlab/SocialiteLlama) and [dataset](https://huggingface.co/datasets/hlab/SocialiteInstructions).

## Instruction tuning

We instruct tune Llama2 7B with the following default hyperparamerters:

| Hyperparameter  | Llama 2 7B |
| ------------- | ------------- |
| LORA_R  | 8  |
| LORA_ALPHA  | 16  |
| LORA_DROPOUT  | 0.05  |
| LORA_TARGET_MODULES  | q_proj, v_proj  |
| BATCH_SIZE  | 64  |
| MICRO_BATCH_SIZE  | 1 |
| LEARNING_RATE  | 1e-4 |
| NUM_EPOCHS  | 5 |

Instruction tuning command:

```
deepspeed --include localhost:0,1 finetuning.py --checkpoint /llama2-7b-hf --dataset hlab/SocialiteInstructions --OUTPUT_DIR /socialite_output_dir
```


## Evaluation

For zero-shot evaluation, `task_type` indicates the task we want to perform the evaluation for.

For example, the command for zero-shot evaluation for `HATESPEECH` is:

```
CUDA_VISIBLE_DEVICES=0 python eval/zeroshot.py --checkpoint hlab/SocialiteLlama --dataset hlab/SocialiteInstructions --output_file /hate_speech_zeroshot_pred_socialite.csv --task_type HATESPEECH
```

The full list of task types can be found in the paper.


## CITE US

```
@inproceedings{
  dey-etal-2024-socialite,
  title={{SOCIALITE}-{LLAMA}: An Instruction-Tuned Model for Social Scientific Tasks},
  author={Dey, Gourab and V Ganesan, Adithya and Lal, Yash Kumar and Shah, Manal and Sinha, Shreyashee and Matero, Matthew and Giorgi, Salvatore and Kulkarni, Vivek and Schwartz, H. Andrew},
  address = "St. Julian’s, Malta",
  booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2024},
  publisher = {Association for Computational Linguistics} 
  }
```
