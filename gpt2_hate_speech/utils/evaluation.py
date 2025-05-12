import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def evaluate(model, val_loader, save_path=None):
    """
    Evaluate the model on the validation set and print the results.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds.extend(outputs["logits"].cpu().numpy())
            truths.extend(batch["labels"].cpu().numpy())

    mse = mean_squared_error(truths, preds)
    mae = mean_absolute_error(truths, preds)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
