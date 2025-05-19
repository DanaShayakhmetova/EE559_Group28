import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import getpass
import os
import random
import re
import tarfile
import time
from pathlib import Path
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torchvision.transforms as transforms
from latent_hate_dataset import LatentHateDataset

from datasets import Dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArgumentss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#Install libraries
#pip install transformers datasets scikit-learn
#Load the data
df = pd.read_csv('implicit_hate_v1_stg1_posts.tsv', sep='\t')

#Check class distribution
print(df['class'].value_counts())

X = df['post']
y = df['class']

#Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])  #Converts Labels to numbers (Here we have 3 different label so e.g 0, 1 or 2)

#Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['post'].tolist(),
    df['label'].tolist(),
    test_size=0.2,    #80% for training, 20% for validation
    random_state=42,
    stratify=df['label']    #Ensures classes are equally represented
)

#Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = LatentHateDataset(train_encodings, train_labels)
val_dataset = LatentHateDataset(val_encodings, val_labels)


#Model
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


#Model loading - Classic BERT with 3 different labels (coherent with our nb of classes)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  
#Defining the training arguments

training_args = TrainingArguments(
    output_dir='./results',
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics = compute_metrics
)
trainer.train()

trainer.evaluate()


#model.save_pretrained('./saved_model_stg1_bert')
#tokenizer.save_pretrained('./saved_model_stg1_bert')

