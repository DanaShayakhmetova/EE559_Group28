import torch 
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
                    classification_metrics,
                    regression_metrics,
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

    student_model = student_model.to(device)
    regression_teacher = regression_teacher.to(device)
    classification_teacher = classification_teacher.to(device)

    regression_teacher.eval()
    classification_teacher.eval()
    student_model.train()

    head_c_loss = 0
    head_r_loss = 0

    head_c_metrics = dict(zip(classification_metrics.keys(), torch.zeros(len(classification_metrics), device=device)))
    head_r_metrics = dict(zip(regression_metrics.keys(), torch.zeros(len(regression_metrics), device=device)))

    n_batches = len(regression_loader) + len(classification_loader)
    assert len(regression_loader) == len(classification_loader) and n_batches == 2 * len(regression_loader), "The two dataloaders should have the same length"

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
            
            r_head_student_out = student_model(input_ids, from_batch, attention_mask=attention_mask)["logits"]

            regression_loss = torch.nn.MSELoss()

            total_loss = alpha * regression_loss(r_teacher_out, labels) + (1 - alpha) * regression_criterion(r_head_student_out, labels)
            head_r_loss += total_loss.item()
        else:
            raise ValueError("The task should either be 'classification' or 'regression'!")
        
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            if from_batch == 'classification':
                c_head_student_out = torch.argmax(c_head_student_out, dim=-1)
                labels = labels.cpu().numpy()
                c_head_student_out = c_head_student_out.cpu().numpy()
                for metric_name, metric_fn in classification_metrics.items():
                    head_c_metrics[metric_name] += metric_fn(c_head_student_out, labels)

            else:
                r_head_student_out = r_head_student_out.squeeze(-1).cpu().numpy()
                labels = labels.cpu().numpy()
                r_head_student_out = r_head_student_out.cpu().numpy()
                for metric_name, metric_fn in regression_metrics.items():
                    head_r_metrics[metric_name] += metric_fn(r_head_student_out, labels)

    
    epoch_c_loss = head_c_loss / len(classification_loader)
    epoch_r_loss = head_r_loss / len(regression_loader)

    for metric_name in head_c_metrics.keys():
        head_c_metrics[metric_name] /= len(classification_loader)

    for metric_name in head_r_metrics.keys():
        head_r_metrics[metric_name] /= len(regression_loader)


    print(f"Classification head loss: {epoch_c_loss:.4f}")
    print(f"Regression head loss: {epoch_r_loss:.4f}")

    return epoch_c_loss, epoch_r_loss


def train(model, 
          regression_teacher, 
          classification_teacher, 
          regression_loader, 
          classification_loader, 
          optimizer, 
          device,
          num_epochs=5,
          alpha=0.5,
          temperature=2.0):
    """
    Train the multi-head student model using two heterogeneous teacher models.
    """
    
    # Define the loss functions
    regression_criterion = torch.nn.MSELoss()
    classification_criterion = torch.nn.CrossEntropyLoss()

    # Define the metrics
    classification_metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

    regression_metrics = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss()
    }
    
    model.to(device)

    print(f"training starting...")
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        c_loss, r_loss = train_one_epoch(model, 
                        regression_teacher, 
                        classification_teacher, 
                        regression_loader, 
                        classification_loader, 
                        regression_criterion, 
                        classification_criterion, 
                        optimizer, 
                        device,
                        classification_metrics,
                        regression_metrics,
                        alpha=alpha,
                        temperature=temperature)
        print(f"Classification head loss: {c_loss:.4f}")
        print(f"Regression head loss: {r_loss:.4f}")

    # Save the model 
    model_dir = "multi_head_student_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer = model.tokenizer
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
    return model



