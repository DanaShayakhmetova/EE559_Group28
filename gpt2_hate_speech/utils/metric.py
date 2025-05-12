import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def metric(model, val_loader, save_path=None):
    """
    Evaluate the model on the validation set and print the results.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            # Predicted class: argmax for multi-class, sigmoid + threshold for binary
            if logits.shape[1] == 1:
                # Binary classification with sigmoid activation
                probs = torch.sigmoid(torch.tensor(logits)).numpy()
                pred_labels = (probs > 0.5).astype(int).flatten()
            else:
                # Multi-class classification
                pred_labels = np.argmax(logits, axis=1)

            preds.extend(pred_labels)
            truths.extend(labels)

    # Convert to numpy arrays
    preds = np.array(preds)
    truths = np.array(truths)

    acc = accuracy_score(truths, preds)
    prec = precision_score(truths, preds, average='macro', zero_division=0)
    rec = recall_score(truths, preds, average='macro', zero_division=0)
    f1 = f1_score(truths, preds, average='macro', zero_division=0)

    results = [
        ("Accuracy", acc),
        ("Precision (macro)", prec),
        ("Recall (macro)", rec),
        ("F1 Score (macro)", f1),
    ]

    ds = pd.DataFrame(results, columns=["Metric", "Score"])
    ds.to_csv(".", index=False)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
