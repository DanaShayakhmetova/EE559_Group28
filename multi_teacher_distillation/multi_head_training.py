import torch 
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
                    classification_criterion,
                    optimizer, 
                    device,
                    classification_metrics,
                    regression_metrics,
                    alpha=0.5,
                    temperature=2.0,
                    epoch=0):
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

    r_loader_size = len(regression_loader)
    c_loader_size = len(classification_loader)
    # Using iterators to loaders of different tasks with different sizes
    classification_iter = iter(classification_loader)
    regression_iter = iter(regression_loader)

    for i in tqdm(range(n_batches), desc="Training", leave=False):
        
        optimizer.zero_grad()
        # Check for the batch source
        if (i % 2 == 0 and c_loader_size != 0) or (r_loader_size == 0):
            data = next(classification_iter)
            from_batch = 'classification'
            c_loader_size -= 1
        elif (i % 2 == 1 and r_loader_size != 0) or (c_loader_size == 0):
            data = next(regression_iter)
            from_batch = 'regression'
            r_loader_size -= 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        epoch_reg_predictions = []
        epoch_reg_labels = []

        if from_batch == 'classification':
            # Classification head training
            with torch.no_grad():
                c_teacher_out = classification_teacher(input_ids, attention_mask=attention_mask)["logits"]
            
            # Convert logits to soft targets
            c_soft_target = torch.softmax(c_teacher_out / temperature, dim=-1)
            c_head_student_out = student_model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
            c_student_prob = torch.log_softmax(c_head_student_out / temperature, dim=-1)

            # Use KL Divergence loss for knowledge distillation because it is usually used for soft targets
            # since it measures the difference between two probability distributions.
            KD_loss = torch.nn.KLDivLoss(reduction='batchmean')

            c_head_loss = KD_loss(c_student_prob, c_soft_target) * (temperature ** 2)
            
            # Compute the classification loss similarly to how it is done in the knowledge distillation paper
            total_loss = alpha * c_head_loss + (1 - alpha) * classification_criterion(c_head_student_out, labels)
            head_c_loss += total_loss.item()
            
        elif from_batch == 'regression':
            # Regression head training
            labels = labels.view(-1)

            with torch.no_grad():
                r_teacher_out = regression_teacher(input_ids, attention_mask=attention_mask)["logits"]
            

            r_head_student_out = student_model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
            r_head_student_out = r_head_student_out.view(-1)

            total_loss = alpha * F.mse_loss(r_head_student_out, r_teacher_out) + (1 - alpha) * F.huber_loss(r_head_student_out, labels)
            head_r_loss += total_loss.item()
        else:
            # In training only two tasks are supported, however in inference mode we allow for getting both heads outputs
            # at the same time.
            raise ValueError("The task should either be 'classification' or 'regression'!")
        
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            if from_batch == 'classification':
                c_head_student_out = torch.argmax(c_head_student_out, dim=-1)
                labels = labels.cpu().numpy()
                c_head_student_out = c_head_student_out.cpu().numpy()

            else:
                r_head_student_out = r_head_student_out.squeeze(-1).cpu().numpy()
                labels = labels.cpu().numpy()

                epoch_reg_predictions.append(r_head_student_out)
                epoch_reg_labels.append(labels)

    #log the results for the epoch
    epoch_c_loss = head_c_loss / len(classification_loader)
    epoch_r_loss = head_r_loss / len(regression_loader)
    epoch_results ={
        "epoch": epoch + 1,
        "classification_loss": epoch_c_loss,
        "regression_loss": epoch_r_loss,
    }
    

    for metric_name, metric_value in head_c_metrics.items():
        epoch_results[f"classification_{metric_name}"] = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
        head_c_metrics[metric_name] /= len(classification_loader)

    for metric_name, metric_value in head_r_metrics.items():
        epoch_results[f"regression_{metric_name}"] = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
        head_r_metrics[metric_name] /= len(regression_loader)

    for metric_name, metric_fn in regression_metrics.items():
        epoch_results[f"regression_(corrected){metric_name}"] = metric_fn(epoch_reg_predictions, epoch_reg_labels)

    print(f"Classification head loss: {epoch_c_loss:.4f}")
    print(f"Regression head loss: {epoch_r_loss:.4f}")

    return epoch_c_loss, epoch_r_loss, epoch_results


def train(model, 
          regression_teacher, 
          classification_teacher, 
          regression_loader, 
          classification_loader, 
          optimizer, 
          device,
          model_dir,
          log_path,
          num_epochs=5,
          alpha=0.5,
          temperature=2.0):
    """
    Train the multi-head student model using two heterogeneous teacher models.
    """

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    classification_criterion = torch.nn.CrossEntropyLoss()

    # Define the metrics for classification and regression tasks they are the same as for BERT model
    classification_metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_pred, y_true: precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': lambda y_pred, y_true: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': lambda y_pred, y_true: f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    # Regression metrics, these are the same as for the Rasch-scale GPT model
    regression_metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error
    }
    model.to(device)

    regression_loss_log = []
    classification_loss_log = []
    all_epoch_results = []

    print(f"training starting...")
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        c_loss, r_loss, epoch_results = train_one_epoch(model, 
                        regression_teacher, 
                        classification_teacher, 
                        regression_loader, 
                        classification_loader, 
                        classification_criterion, 
                        optimizer, 
                        device,
                        classification_metrics,
                        regression_metrics,
                        alpha=alpha,
                        temperature=temperature,
                        epoch=epoch)
        print(f"Classification head loss: {c_loss:.4f}")
        print(f"Regression head loss: {r_loss:.4f}")
        classification_loss_log.append(c_loss)
        regression_loss_log.append(r_loss)
        all_epoch_results.append(epoch_results)

        pd.DataFrame(all_epoch_results).to_csv(log_path + f"alpha={alpha}_multi_head_training_results.csv", index=False)
        pd.DataFrame(classification_loss_log).to_csv(log_path + f"alpha={alpha}_classification_train_loss.csv", index=False)
        pd.DataFrame(regression_loss_log).to_csv(log_path + f"alpha={alpha}_regression_train_loss.csv", index=False)

    # Save the model 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(model_dir + f'alpha={alpha}_weights')
    tokenizer = model.tokenizer
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
    return model



