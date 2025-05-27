import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from transformers import DebertaConfig, DebertaTokenizer, get_scheduler
from deberta_rasch_scale.evaluate_deberta import run_metric_evaluation_deberta  
from deberta_rasch_scale.deberta_rasch_scale_model import DebertaRegression    
from gpt2_hate_speech.hate_speech_dataset import HateSpeechDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

num_epoch = 1
test_size = 0.99

dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
df = dataset['train'].to_pandas()
df = df[["text", "hate_speech_score"]].dropna()

train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=test_size, random_state=42)

train_text, train_labels = train_df["text"].tolist(), train_df["hate_speech_score"].tolist()
val_text, val_labels = valid_df["text"].tolist(), valid_df["hate_speech_score"].tolist()
test_text, test_labels = test_df["text"].tolist(), test_df["hate_speech_score"].tolist()

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

train_dataset = HateSpeechDataset(train_text, train_labels, tokenizer)
val_dataset = HateSpeechDataset(val_text, val_labels, tokenizer)
test_dataset = HateSpeechDataset(test_text, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

config = DebertaConfig.from_pretrained("microsoft/deberta-base")
model = DebertaRegression.from_pretrained("microsoft/deberta-base", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=5,
    num_training_steps=len(train_loader) * 5,
)

metric = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}

print(f"Training DeBERTa on {device}")
all_epoch_results = []
loss_at_epoch = []

for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    epoch_predictions = []
    epoch_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step() 

        optimizer.zero_grad()

        total_loss += loss.item()

        with torch.no_grad():
            pred = outputs["logits"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            epoch_predictions.extend(pred)
            epoch_labels.extend(labels)

    epoch_result = {"epoch": epoch + 1, "loss": total_loss / len(train_loader)}
    for metric_name, metric_fn in metric.items():
        epoch_result[metric_name] = metric_fn(epoch_predictions, epoch_labels)

    all_epoch_results.append(epoch_result)
    loss_at_epoch.append(total_loss / len(train_loader))

    pd.DataFrame(all_epoch_results).to_csv("deberta_epoch_results.csv", index=False)

save_dir = "./deberta-base-regression-finetuned"
model.save_weights(save_dir)
tokenizer.save_pretrained(save_dir)
pd.DataFrame(loss_at_epoch, columns=["loss"]).to_csv("deberta_training_loss.csv", index=False)

run_metric_evaluation_deberta(
    weights_path=save_dir,
    test_loader=val_loader,
    metric=metric,
    num_epoch=num_epoch,
    log_path="./eval-logs-deberta/",
    device=device,
)
