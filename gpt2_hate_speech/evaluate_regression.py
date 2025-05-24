from .gpt2_regression_model import GPT2Regression
import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
from .hate_speech_dataset import HateSpeechDataset
from tqdm import tqdm
import os


def run_metric_evalution(weights_path,
                         test_loader,
                         metric,
                         device='cpu',
                         log_path=None,
                         num_epoch=1,
                         ):
    """
    Run classification metric evaluation on the test set.

    Args:

        weights_path (str): Path to the model weights.  

        test_loader (DataLoader): DataLoader for the test set.

        metric (dict): Dictionary of metrics to evaluate.

    """

    save_path = log_path if log_path != None else "./evaluation_gpt2_regression/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
    df = dataset['train'].to_pandas()

    df = df[["text", "hate_speech_score"]].dropna()
    df["label"] = (df["hate_speech_score"] > 0.5).astype(int)

    tokenizer, config, model = GPT2Regression.load_model(weight_path=weights_path)

    model.to(device)
    model.eval()

    all_epoch_results = []
    print(f"start evaluation")
    for epoch in range(num_epoch):
        
        epoch_predictions = []
        epoch_labels = []

        epoch_loss = 0

        epoch_results = {
            "epoch": epoch + 1
            }

        for data in tqdm(test_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in data.items()}
            labels = data['labels'].to(device)

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                epoch_loss += loss.item()

                labels = labels.cpu().numpy()
                logits = logits.cpu().numpy()

                epoch_predictions.append(logits)
                epoch_labels.append(labels)


        for metric_name, metric_fn in metric.items():
            epoch_results[metric_name] = metric_fn(epoch_predictions, epoch_labels)
        all_epoch_results.append(epoch_results)
        
        
        pd.DataFrame(all_epoch_results).to_csv(log_path + "epoch_results.csv", index=False)

