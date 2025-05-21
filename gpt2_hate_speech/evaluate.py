import gpt2_regression_model as gpt2_reg
import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
from utils.metric import metric 
from hate_speech_dataset import HateSpeechDataset


def run_metric_evalution():
    """
    Run classification metric evaluation on the test set.
    """
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
    df = dataset['train'].to_pandas()

    df = df[["text", "hate_speech_score"]].dropna()
    df["label"] = (df["hate_speech_score"] > 0.5).astype(int)

    seed = 42

    tokenizer_small, _, small_model = gpt2_reg.GPT2Regression.load_model("./gpt2-small-regression-finetuned")
    tokenizer_large, _, large_model = gpt2_reg.GPT2Regression.load_model("./gpt2-large-regression-finetuned")

    _, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    small_model.to(device)
    large_model.to(device)

    test_text = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    # tokenizer_small.pad_token = tokenizer_small.eos_token
    # tokenizer_large.pad_token = tokenizer_large.eos_token

    test_dataset = HateSpeechDataset(test_text, test_labels, tokenizer_small)
    test_loader_small = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    test_dataset_large = HateSpeechDataset(test_text, test_labels, tokenizer_large)
    test_loader_large = torch.utils.data.DataLoader(test_dataset_large, batch_size=4)

    print("Evaluating small model...")
    metric(small_model, test_loader_small)

    print("Evaluating large model...")
    metric(large_model, test_loader_large)

run_metric_evalution()