import os

os.system('pip install nltk rouge_score')
os.system('pip install accelerate nvidia-ml-py3')

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate
from transformers import Trainer
import torch
import pandas as pd
from pynvml import *
from transformers import set_seed

set_seed(14)
# Configurations
batch_size = 1
epochs=1

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["completion"], padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py").to("cuda")
trained_model = T5ForConditionalGeneration.from_pretrained(f"folder_path-{batch_size}-ep-{epochs}").to("cuda")
instructions = pd.read_csv("Data/Job_Shop_Scheduling/instructions.txt", sep=" ", header=None) # Change accordingly to
# the data set
instructions = instructions[0].astype(str).values.tolist()
test_data = pd.read_csv('file_path')
input_sequences = test_data["prompt"].astype(str).values.tolist()
size = len(test_data)
generated_code = []

for j in range(size):
    problem_description = input_sequences[j]
    completion = ""
    for i in instructions:
        prompt = problem_description + i
        input_ids = (tokenizer(prompt, return_tensors="pt").input_ids).to("cuda")
        outputs = trained_model.generate(input_ids, max_length=600)
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # code = "test"
        completion = completion + "\n"
        completion = completion + code
    generated_code.append(completion)

test_data["generated_code"] = generated_code
test_data.to_csv(f'file_path_{batch_size}_ep_{epochs}.csv', index=False)