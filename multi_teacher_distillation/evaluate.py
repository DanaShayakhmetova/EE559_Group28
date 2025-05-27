import torch
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

def evaluate_one_epoch(model,
             classification_criterion,
             regression_criterion,
             test_loader_c,
             classification_metrics,
             regression_metrics,
             test_loader_r,
             device,
             epoch=0):
    """
    Evaluate the two heads of the model on the test set.
    Both heads are evaluated on their respective test set.
    """

    model.eval()

    r_epoch_loss = 0
    c_epoch_loss = 0

    c_loader_size = len(test_loader_c)
    r_loader_size = len(test_loader_r)
    regression_loader = iter(test_loader_r)
    classification_loader = iter(test_loader_c)

    n_batches = len(test_loader_r) + len(test_loader_c)

    epoch_reg_predictions = []
    epoch_reg_labels = []

    epoch_class_preds = []
    epoch_class_labels = []

    for i in tqdm(range(n_batches), desc="Evaluating", leave=False):
        if (i % 2 == 0 and c_loader_size != 0) or (r_loader_size == 0):
            data = next(classification_loader)
            from_batch = 'classification'
            c_loader_size -= 1
        elif (i % 2 == 1 and r_loader_size != 0) or (c_loader_size == 0):
            data = next(regression_loader)
            from_batch = 'regression'
            r_loader_size -= 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        with torch.no_grad():
            if from_batch == 'classification':
                c_head_student_out = model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
                
                c_head_loss = classification_criterion(c_head_student_out, labels)
                c_epoch_loss += c_head_loss.item()
                c_head_student_out = torch.argmax(c_head_student_out, dim=-1)
                
                labels = labels.cpu().numpy()
                c_head_student_out = c_head_student_out.cpu().numpy()

                epoch_class_preds.append(c_head_student_out)
                epoch_class_labels.append(labels)

            elif from_batch == 'regression':
                r_head_student_out = model(input_ids, from_batch, attention_mask=attention_mask)["logits"]  # shape [B, 1]

                r_head_loss = regression_criterion(r_head_student_out, labels)
                r_epoch_loss += r_head_loss.item()

                # Flatten both to [B]
                labels = labels.view(-1).cpu().numpy()
                r_head_student_out = r_head_student_out.view(-1).cpu().numpy()

                print(f"r_head_student_out: {r_head_student_out}, labels: {labels}")

                epoch_reg_predictions.extend(r_head_student_out.tolist())
                epoch_reg_labels.extend(labels.tolist())
            else:
                raise ValueError("The task should either be 'classification' or 'regression'!")
            
    epoch_c_loss = c_epoch_loss / len(test_loader_c)
    epoch_r_loss = r_epoch_loss / len(test_loader_r)

    epoch_results = {
        "epoch": epoch + 1,
        "classification_loss": epoch_c_loss,
        "regression_loss": epoch_r_loss,
    }

    concatenated_class_preds = np.concatenate(epoch_class_preds)
    concatenated_class_labels = np.concatenate(epoch_class_labels)
    
    for metric_name, metric_fn in classification_metrics.items():
        epoch_results[f"classification_{metric_name}"] = metric_fn(concatenated_class_preds, concatenated_class_labels)

    for metric_name, metric_fn in regression_metrics.items():
        epoch_results[f"regression_{metric_name}"] = metric_fn(epoch_reg_predictions, epoch_reg_labels)

    return epoch_c_loss, epoch_r_loss, epoch_results


def evaluate(model, 
            regression_criterion,
            classification_criterion,
            num_epochs,
            test_loader_c,
            test_loader_r,
            classification_metrics,
            regression_metrics,
            eval_path,
            device):
    """
    Evaluate the two heads of the model on the test set.
    Both heads are evaluated on their respective test set.
    """

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    regression_loss_log = []
    classification_loss_log = []
    all_epoch_results = []

    print(f"Evaluating the model...")
    for epoch in tqdm(range(num_epochs), desc="Evaluating", leave=False):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        classification_loss, regression_loss, epoch_results = evaluate_one_epoch(
            model=model,
            regression_criterion=regression_criterion,
            classification_criterion=classification_criterion,
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            test_loader_c=test_loader_c,
            test_loader_r=test_loader_r,
            device=device,
            epoch=epoch
        )
        classification_loss_log.append(classification_loss)
        regression_loss_log.append(regression_loss)

        all_epoch_results.append(epoch_results)

        pd.DataFrame(all_epoch_results).to_csv(eval_path + "all_eval_metrics.csv", index=False)
        pd.DataFrame(classification_loss_log).to_csv(eval_path + "classification_eval_loss.csv", index=False)
        pd.DataFrame(regression_loss_log).to_csv(eval_path + "regression_eval_loss.csv", index=False)

    return model

