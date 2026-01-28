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

dataset = load_dataset("csv", data_files="file_path")
test_dataset = load_dataset("csv", data_files="jfile_path")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")

def tokenize_function(examples):
    model_inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["completion"], padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"]
small_test_dataset = tokenized_test_datasets["train"]

model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py").to("cuda")

training_args = Seq2SeqTrainingArguments(output_dir="folder_path",
                                         do_eval="True",
                                         evaluation_strategy = "steps",
                                         eval_steps=10,
                                         logging_steps=10,
                                         save_steps=50,
                                         save_strategy="steps",
                                         per_device_train_batch_size=batch_size,
                                         gradient_checkpointing=True,
                                         save_total_limit=3,
                                         num_train_epochs=epochs
                                         )

data_collator = DataCollatorForSeq2Seq(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
result = trainer.train()
detailed_result = trainer.state.log_history
pd.DataFrame([str(result), str(detailed_result)]).to_csv(f'file_path-{batch_size}-ep-{epochs}.csv', index=False)
trainer.save_model(f"folder_path-{batch_size}-ep-{epochs}")