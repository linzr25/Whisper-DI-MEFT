import os
from huggingface_hub import login

import sys
sys.path.append('../')

from transformers import (
    WhisperFeatureExtractor,
    Trainer,
    TrainingArguments,
)
from whisper_modified.whisper_for_DI import WhisperForAudioClassification
from datasets import DatasetDict, load_from_disk, concatenate_datasets
import torch
from torch import nn
from sklearn.metrics import accuracy_score
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union


label2id = {
    "Ji-Lu": 0,
    "Jiang-Huai": 1,
    "Jiao-Liao": 2,
    "Lan-Yin": 3,
    "Southwestern": 4,
    "Zhongyuan": 5
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
print(num_labels)


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

dataset = DatasetDict()

dataset["train"] = load_from_disk("../Datasets/KeSpeech_Dialects/prepared_full/train")
dataset["test"] = load_from_disk("../Datasets/KeSpeech_Dialects/prepared_full/test")


# @dataclass
# class DataCollatorForClassification:
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         labels = [feature["labels"] for feature in features]
        
#         batch = feature_extractor.pad(input_features, return_tensors="pt")
        
#         batch["labels"] = torch.tensor(labels)
#         return batch

# data_collator = DataCollatorForClassification()

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    num_dialects = len(label2id)

    overall_accuracy = accuracy_score(labels, preds)

    dialect_accuracies = {}
    for dialect_name, dialect_id in label2id.items():
        dialect_mask = labels == dialect_id
        if np.sum(dialect_mask) > 0:
            dialect_acc = accuracy_score(labels[dialect_mask], preds[dialect_mask])
            dialect_accuracies[f"accuracy_{dialect_name}"] = dialect_acc
        else:
            dialect_accuracies[f"accuracy_{dialect_name}"] = 0.0
    
    average_accuracy = np.mean(list(dialect_accuracies.values()))

    return {
        "accuracy_overall": overall_accuracy,
        "average_accuracy": average_accuracy, 
        **dialect_accuracies,
    }

model = WhisperForAudioClassification.from_pretrained(
    "openai/whisper-small",
    num_labels=num_labels,
    classifier_proj_size=256,  
    use_weighted_layer_sum=False, 
    ignore_mismatched_sizes=True,
)

for name, param in model.encoder.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_params = 0
total_params = 0

print("Trainable layers:")
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        print(f"  {name}: {param.numel()} parameters")
        trainable_params += param.numel()

print(f"\nSummary:")
print(f"Trainable params: {trainable_params} / Total params: {total_params} ({100 * trainable_params / total_params:.2f}%)")


training_args = TrainingArguments(
    output_dir="../whisper-small-DI-BitFit", 
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1, 
    eval_accumulation_steps=1,
    learning_rate=1e-4,
    num_train_epochs=10,
    # gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="accuracy_overall",
    greater_is_better=True,
    dataloader_num_workers=64,  
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()