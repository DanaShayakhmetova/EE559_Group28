import torch
from tqdm import tqdm
import pandas as pd

def evaluate_one_epoch(model,
             classification_criterion,
             regression_criterion,
             metrics,
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

    r_head_metrics = dict(zip(regression_metrics.keys(), torch.zeros(len(regression_metrics), device=device)))
    c_head_metrics = dict(zip(classification_metrics.keys(), torch.zeros(len(classification_metrics), device=device)))

    regression_loader = iter(test_loader_r)
    classification_loader = iter(test_loader_c)
    n_batches = len(test_loader_r) + len(test_loader_c)

    assert len(test_loader_r) == len(test_loader_c) and n_batches == 2 * len(test_loader_r), "The two dataloaders should have the same length"

    for i in tqdm(range(n_batches), desc="Evaluating", leave=False):
        if i % 2 == 0:
            data = next(classification_loader)
            from_batch = 'classification'
        else:
            data = next(regression_loader)
            from_batch = 'regression'

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
                for metric_name, metric_fn in metrics.items():
                    c_head_metrics[metric_name] += metric_fn(c_head_student_out, labels)

            elif from_batch == 'regression':
                r_head_student_out = model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
                labels = labels.view(-1)

                r_head_loss = regression_criterion(r_head_student_out, labels)
                r_epoch_loss += r_head_loss.item()
                
                labels = labels.cpu().numpy()
                r_head_student_out = r_head_student_out.cpu().numpy()
                
                for metric_name, metric_fn in metrics.items():
                    r_head_metrics[metric_name] += metric_fn(r_head_student_out, labels)
            else:
                raise ValueError("The task should either be 'classification' or 'regression'!")
            
    epoch_c_loss = c_epoch_loss / len(test_loader_c)
    epoch_r_loss = r_epoch_loss / len(test_loader_r)

    for metric_name in c_head_metrics.keys():
        c_head_metrics[metric_name] /= len(test_loader_c)
    
    for metric_name in r_head_metrics.keys():
        r_head_metrics[metric_name] /= len(test_loader_r)

    print(f"Classification head loss: {epoch_c_loss:.4f}")
    print(f"Regression head loss: {epoch_r_loss:.4f}")
    epoch_results = {
        "epoch": epoch + 1,
        "classification_loss": epoch_c_loss,
        "regression_loss": epoch_r_loss,
    }
    for metric_name in c_head_metrics.keys():
        epoch_results[f"classification_{metric_name}"] = c_head_metrics[metric_name].item() if isinstance(c_head_metrics[metric_name], torch.Tensor) else c_head_metrics[metric_name]
        print(f"Classification {metric_name}: {c_head_metrics[metric_name]:.4f}")
    
    for metric_name in r_head_metrics.keys():
        epoch_results[f"regression_{metric_name}"] = r_head_metrics[metric_name].item() if isinstance(r_head_metrics[metric_name], torch.Tensor) else r_head_metrics[metric_name]
        print(f"Regression {metric_name}: {r_head_metrics[metric_name]:.4f}")
    

    return epoch_c_loss, epoch_r_loss


def evaluate(model, 
            regression_criterion,
            classification_criterion,
            num_epochs,
            test_loader_c,
            test_loader_r,
            classification_metrics,
            regression_metrics,
            device):
    """
    Evaluate the two heads of the model on the test set.
    Both heads are evaluated on their respective test set.
    """

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

        pd.DataFrame(all_epoch_results).to_csv("all_eval_metrics.csv", index=False)
        pd.DataFrame(classification_loss_log).to_csv("classification_eval_loss.csv", index=False)
        pd.DataFrame(regression_loss_log).to_csv("regression_eval_loss.csv", index=False)

    return model

