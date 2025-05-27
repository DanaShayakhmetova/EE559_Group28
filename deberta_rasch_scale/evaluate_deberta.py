from .deberta_rasch_scale_model import DebertaRegression
import pandas as pd
import torch
from datasets import load_dataset
from transformers import DebertaTokenizer
from sklearn.model_selection import train_test_split
from gpt2_hate_speech.hate_speech_dataset import HateSpeechDataset
from tqdm import tqdm
import os


def run_metric_evaluation_deberta(weights_path,
                                 test_loader,
                                 metric,
                                 device='cpu',
                                 log_path=None,
                                 num_epoch=1):
    """
    Run regression metric evaluation on the test set using DeBERTa model.

    Args:
        weights_path (str): Path to the model weights.  
        test_loader (DataLoader): DataLoader for the test set.
        metric (dict): Dictionary of metrics to evaluate (e.g., MAE, MSE).
        device (str): Device to run the evaluation on.
        log_path (str): Optional path to save logs/results.
        num_epoch (int): Number of passes through the test set (useful if using dropout).
    """

    save_path = log_path if log_path is not None else "./evaluation_deberta_regression/"
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
    df = dataset['train'].to_pandas()

    df = df[["text", "hate_speech_score"]].dropna()
    df["label"] = (df["hate_speech_score"] > 0.5).astype(int)

    tokenizer, config, model = DebertaRegression.load_model(weight_path=weights_path)
    model.to(device)
    model.eval()

    all_epoch_results = []
    print(f"Start DeBERTa evaluation")

    for epoch in range(num_epoch):
        epoch_predictions = []
        epoch_labels = []
        epoch_loss = 0

        epoch_results = {"epoch": epoch + 1}

        for data in tqdm(test_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in data.items()}
            labels = batch['labels']

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
                logits = outputs["logits"]

                epoch_loss += loss.item()
                epoch_predictions.extend(logits.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        for metric_name, metric_fn in metric.items():
            epoch_results[metric_name] = metric_fn(epoch_predictions, epoch_labels)
        all_epoch_results.append(epoch_results)

        pd.DataFrame(all_epoch_results).to_csv(os.path.join(save_path, "epoch_results.csv"), index=False)
