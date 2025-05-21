import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import multi_head_training
import multi_head_student
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bert_latent_hatred.latent_hate_dataset import LatentHateDataset
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from gpt2_hate_speech.hate_speech_dataset import HateSpeechDataset
from gpt2_hate_speech.gpt2_regression_model import GPT2Regression
import pandas as pd
from evaluate import evaluate


def train_multi_head():
    print(f"Trying to load the models")
    PATH = os.path.abspath("./model_weights/gpt2_large_regression_finetuned")
    _, _, regression_teacher = GPT2Regression.load_model(PATH, local_files_only=True)
    classification_teacher = BertForSequenceClassification.from_pretrained("./model_weights/saved_model_stg1_bert")
    print(f"model loaded successfully")

    print(f"start loading datasets")
    regression_dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
    df_regression = regression_dataset['train'].to_pandas()

    df_regression = df_regression[["text", "hate_speech_score"]].dropna()
    df_clustering = pd.read_csv('../implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv', sep='\t')

    X = df_clustering['post']
    y = df_clustering['class']

    le = LabelEncoder()
    df_clustering['label'] = le.fit_transform(df_clustering['class']) 

    train_regression, temp_regression = train_test_split(df_regression, test_size=0.2, random_state=42)

    train_reg_texts = train_regression['text'].tolist()
    train_reg_labels = train_regression['hate_speech_score'].tolist()

    val_reg_texts = temp_regression['text'].tolist()
    val_reg_labels = temp_regression['hate_speech_score'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_clustering['post'].tolist(),
        df_clustering['label'].tolist(),
        test_size=0.2,  
        random_state=42,
        stratify=df_clustering['label']   
    )

    print(f"loading of datasets and data preparation done successfully")

    print(f"student model creation...")
    m_head_student = multi_head_student.MultiHeadStudent(
        pretrained_model_name='bert-base-uncased',
        num_classes=3, 
        use_pooler=False,
        use_activation=False,
        activation_function=torch.nn.ReLU,
        use_dropout=False,
        dropout_rate=0.1,
    )

    tokenizer = m_head_student.tokenizer

    train_clustering_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_clustering_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_clustering_dataset = LatentHateDataset(train_clustering_encodings, train_labels)
    val_clustering_dataset = LatentHateDataset(val_clustering_encodings, val_labels)

    train_reg_encoding =tokenizer(train_reg_texts, truncation=True, padding=True, max_length=128)
    val_reg_encoding =tokenizer(val_reg_texts, truncation=True, padding=True, max_length=128)

    train_reg_dataset = HateSpeechDataset(train_reg_texts, train_reg_labels, tokenizer)
    val_reg_dataset = HateSpeechDataset(val_reg_texts, val_reg_labels, tokenizer)

    smallest_dataset = min(len(train_clustering_dataset), len(train_reg_dataset))
    train_clustering_dataset = Subset(train_clustering_dataset, range(smallest_dataset))
    train_reg_dataset = Subset(train_reg_dataset, range(smallest_dataset))

    smallest_val_dataset = min(len(val_clustering_dataset), len(val_reg_dataset))
    val_clustering_dataset = Subset(val_clustering_dataset, range(smallest_val_dataset))
    val_reg_dataset = Subset(val_reg_dataset, range(smallest_val_dataset))

    clustering_train_loader = DataLoader(train_clustering_dataset, batch_size=4, shuffle=True)
    clustering_val_loader = DataLoader(val_clustering_dataset, batch_size=4, shuffle=False)

    regression_train_loader = DataLoader(train_reg_dataset, batch_size=4, shuffle=True)
    regression_val_loader = DataLoader(val_reg_dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.AdamW(m_head_student.parameters(), lr=5e-5)

    print(f"training starting...")
    trained_model = multi_head_training.train(
        model=m_head_student, 
        regression_teacher=regression_teacher,
        classification_teacher=classification_teacher,
        classification_loader=clustering_train_loader,
        regression_loader=regression_train_loader,
        num_epochs=5,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return trained_model
