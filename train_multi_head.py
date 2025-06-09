import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import multi_teacher_distillation.multi_head_training as multihead_training
import multi_teacher_distillation.multi_head_student as multihead_student
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
from multi_teacher_distillation.evaluate import evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Run this script mulitple times on with different GPUs to test multiple seeds simultaneously, use one GPU per run (5 seeds should be enough) 
# Then recover the results saved in the CSV files

# Set the parameters for training 

# Number of epochs, test sizes, and model name (small, medium, large) for the GPT-2 model
num_epochs = 1
reg_test_size = 0.99
class_test_size = 0.99

validation_split_reg = 0.99
validation_split_class = 0.99
model_name = "small"

print(f"Trying to load the models")
# Load the GPT-2 regression model and the BERT classification model
path = f"./gpt2-{model_name}-regression-finetuned-fixed-smaller-train"
_, _, regression_teacher = GPT2Regression.load_model(path, local_files_only=True)
classification_teacher = BertForSequenceClassification.from_pretrained("./saved_model_stg1_bert")
print(f"model gpt2-{model_name} loaded successfully")

print(f"start loading datasets")
regression_dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
df_regression = regression_dataset['train'].to_pandas()

df_regression = df_regression[["text", "hate_speech_score"]].dropna()
df_clustering = pd.read_csv('./implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv', sep='\t')

X = df_clustering['post']
y = df_clustering['class']

le = LabelEncoder()
df_clustering['label'] = le.fit_transform(df_clustering['class']) 

# Add a random state if desired for reproducibility
train_regression, temp_regression = train_test_split(df_regression, test_size=reg_test_size)
temp_regression, _ = train_test_split(temp_regression, test_size=validation_split_reg)

train_reg_texts = train_regression['text'].tolist()
train_reg_labels = train_regression['hate_speech_score'].tolist()

val_reg_texts = temp_regression['text'].tolist()
val_reg_labels = temp_regression['hate_speech_score'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_clustering['post'].tolist(),
    df_clustering['label'].tolist(),
    test_size=class_test_size,
    stratify=df_clustering['label']
)

print(f"loading of datasets and data preparation done successfully")

print(f"student model creation...")
bert_model_name = 'bert-base-uncased'  # or 'roberta-base', 'distilbert-base-uncased', etc.

m_head_student = multihead_student.MultiHeadStudent(
    pretrained_model_name=bert_model_name,
    num_classes=3,
    use_pooler=False,
    use_activation=False,
    activation_function=torch.nn.ReLU,
    use_dropout=False,
    dropout_rate=0.1,
    roberta=False
)

tokenizer = m_head_student.tokenizer

train_clustering_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_clustering_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_clustering_dataset = LatentHateDataset(train_clustering_encodings, train_labels)
val_clustering_dataset = LatentHateDataset(val_clustering_encodings, val_labels)

train_reg_dataset = HateSpeechDataset(train_reg_texts, train_reg_labels, tokenizer)
val_reg_dataset = HateSpeechDataset(val_reg_texts, val_reg_labels, tokenizer)

clustering_train_loader = DataLoader(train_clustering_dataset, batch_size=1, shuffle=True)
clustering_val_loader = DataLoader(val_clustering_dataset, batch_size=1, shuffle=False)

regression_train_loader = DataLoader(train_reg_dataset, batch_size=1, shuffle=True)
regression_val_loader = DataLoader(val_reg_dataset, batch_size=1, shuffle=False)

optimizer = torch.optim.AdamW(m_head_student.parameters(), lr=5e-5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"training starting...")
# Change the alpha values to test different weights for the classification and regression losses
alphas = [0.5]

for alpha in alphas:

    trained_model = multihead_training.train(
        model=m_head_student, 
        regression_teacher=regression_teacher,
        classification_teacher=classification_teacher,
        classification_loader=clustering_train_loader,
        regression_loader=regression_train_loader,
        num_epochs=num_epochs,
        model_dir=f"./{model_name}_alpha={alpha}_{bert_model_name}_multi_head_model/",
        log_path=f"./{model_name}_alpha={alpha}_{bert_model_name}_multi_head_model_train_logs/",
        optimizer=optimizer,
        alpha=alpha,
        device=device
    )

    classification_metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_pred, y_true: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': lambda y_pred, y_true: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': lambda y_pred, y_true: f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    regression_metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error
    }

    evaluate(
        model=trained_model, 
        regression_criterion=torch.nn.MSELoss(),
        classification_criterion=torch.nn.CrossEntropyLoss(),
        num_epochs=num_epochs,
        test_loader_c=clustering_val_loader,
        test_loader_r=regression_val_loader,
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
        eval_path=f"./{model_name}_alpha={alpha}_{bert_model_name}_multi_head_model_eval_logs/",
        device=device
    )