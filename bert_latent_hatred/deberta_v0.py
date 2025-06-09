import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#load the dataset from the latent hatred paper - stg1 posts
df = pd.read_csv('implicit_hate_v1_stg1_posts.tsv', sep='\t')

#label encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

# Split into train (70%) + validation (15%) + test (15%)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['post'].tolist(),
    df['label'].tolist(),
    test_size=0.3,
    random_state=42,
    stratify=df['label']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

#
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)

#tokenization
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        } | {
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

train_dataset = HateDataset(train_encodings, train_labels)
val_dataset = HateDataset(val_encodings, val_labels)
test_dataset = HateDataset(test_encodings, test_labels)

#choice of models - 3 labels for classification {no hate, explicit hate, implicit hate}
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base',
    num_labels=3   
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

#training arguments
training_args = TrainingArguments(
    output_dir='./results_deberta',
    eval_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    save_total_limit=1,
)

#trainer method
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

#training
trainer.train()

#evaluation on validation set
val_results = trainer.evaluate()
print("Validation Results:")
print(val_results)

#saving
with open("eval_metrics_DEBERTA_val.txt", "w") as f:
    for key, value in val_results.items():
        f.write(f"{key}: {value}\n")

#evaluation on the test set
test_results = trainer.evaluate(test_dataset)
print("Test Results:")
print(test_results)

#saving
with open("eval_metrics_DEBERTA_test.txt", "w") as f:
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")

# Save model and tokenizer
model.save_pretrained('./saved_model_stg1_DeBERTa')
tokenizer.save_pretrained('./saved_model_stg1_DeBERTa')