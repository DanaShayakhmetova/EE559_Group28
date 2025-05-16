import torch 
from multi_head_student import MultiHeadStudent
from gpt2_hate_speech import GPT2HateSpeech
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os



def train_one_epoch(student_model,
                    regression_teacher,  
                    classification_teacher,  
                    regression_loader,
                    classification_loader, 
                    regression_criterion,
                    classification_criterion,
                    optimizer, 
                    device,
                    metrics,
                    alpha=0.5,
                    temperature=2.0):
    """
    Train the multi-head student model for one epoch using heterogeneous teacher models to distill knowledge.
    The student model has two heads, one for classification and one for regression.

    The idea is to use a mixed batch, train the encoder during every steps but only updating 
    one head at a time. This should allow the encoder from the two teacher models to recognize
    hate speech and the rasch score. This way the encoder should have learnt the embedding space
    for both tasks which are similar.

    When training the heads, we use a similar approach to the original knowledges distillation paper
    (https://arxiv.org/abs/1503.02531) and use the logits from the two teacher models to compute the soft targets.
    """

    regression_teacher.eval()
    classification_teacher.eval()
    student_model.train()

    head_c_loss = 0
    head_r_loss = 0

    head_c_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics), device=device)))
    head_r_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics), device=device)))

    assert len(regression_loader) == len(classification_loader), "The two dataloaders should have the same length"

    n_batches = len(regression_loader) + len(classification_loader)

    classification_iter = iter(classification_loader)
    regression_iter = iter(regression_loader)

    for i in tqdm(range(n_batches), desc="Training", leave=False):
        
        optimizer.zero_grad()
        
        if i % 2 == 0:
            data = next(classification_iter)
            from_batch = 'classification'
        else:
            data = next(regression_iter)
            from_batch = 'regression'

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        if from_batch == 'classification':
            with torch.no_grad():
                # Three classes teacher classifier
                c_teacher_out = classification_teacher(input_ids, attention_mask=attention_mask)["logits"]
            
            c_soft_target = torch.softmax(c_teacher_out / temperature, dim=-1)
            c_head_student_out = student_model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
            c_student_prob = torch.log_softmax(c_head_student_out / temperature, dim=-1)

            KD_loss = torch.nn.KLDivLoss(reduction='batchmean')
            c_head_loss = KD_loss(c_student_prob, c_soft_target) * (temperature ** 2)
            
            total_loss = alpha * c_head_loss + (1 - alpha) * classification_criterion(c_head_student_out, labels)
            head_c_loss += total_loss.item()
            
        elif from_batch == 'regression':
            with torch.no_grad():
                # Regression teacher's output 
                r_teacher_out = regression_teacher(input_ids, attention_mask=attention_mask)["logits"]
            
            r_soft_target = torch.softmax(r_teacher_out)
            r_head_student_out = student_model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
            r_student_prob = torch.log_softmax(r_head_student_out)
            r_head_loss = torch.nn.KLDivLoss(reduction='batchmean')
            r_head_loss = KD_loss(r_student_prob, r_soft_target) * (temperature ** 2)
            total_loss = alpha * r_head_loss + (1 - alpha) * regression_criterion(r_head_student_out, labels)
            head_r_loss += total_loss.item()

        else:
            raise ValueError("The task should either be 'classification' or 'regression'!")
        
        total_loss.backward()
        optimizer.step()
        for metric_name, metric_fn in metrics.items():
            if from_batch == 'classification':
                head_c_metrics[metric_name] += metric_fn(c_head_student_out, labels)
            else:
                head_r_metrics[metric_name] += metric_fn(r_head_student_out, labels)
            

    pass