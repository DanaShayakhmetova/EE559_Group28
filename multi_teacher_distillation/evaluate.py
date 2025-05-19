import torch
from tqdm import tqdm

def evaluate(model,
             criterion,
             metrics,
             test_loader_c,
             test_loader_r,
             device):
    """
    Evaluate the two heads of the model on the test set.
    Both heads are evaluated on their respective test set.
    """

    model.eval()

    r_epoch_loss = 0
    c_epoch_loss = 0

    r_head_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics), device=device)))
    c_head_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics), device=device)))

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
                c_head_loss = criterion(c_head_student_out, labels)
                c_epoch_loss += c_head_loss.item()
                c_head_student_out = torch.argmax(c_head_student_out, dim=-1)
                labels = labels.cpu().numpy()
                c_head_student_out = c_head_student_out.cpu().numpy()
                for metric_name, metric_fn in metrics.items():
                    c_head_metrics[metric_name] += metric_fn(c_head_student_out, labels)

            elif from_batch == 'regression':
                r_head_student_out = model(input_ids, from_batch, attention_mask=attention_mask)["logits"]
                r_head_loss = criterion(r_head_student_out, labels)
                r_epoch_loss += r_head_loss.item()
                r_head_student_out = torch.argmax(r_head_student_out, dim=-1)
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
    for metric_name in c_head_metrics.keys():
        print(f"Classification {metric_name}: {c_head_metrics[metric_name]:.4f}")
    
    for metric_name in r_head_metrics.keys():
        print(f"Regression {metric_name}: {r_head_metrics[metric_name]:.4f}")
    

    return epoch_c_loss, epoch_r_loss

