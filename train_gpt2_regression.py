import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GPT2Config
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, get_scheduler
from gpt2_hate_speech.evaluate_regression import run_metric_evalution

from gpt2_hate_speech.gpt2_regression_model import GPT2Regression
from gpt2_hate_speech.hate_speech_dataset import HateSpeechDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# These values are like that only for the sake of this video
train_size = 0.99
test_size = 0.99

batch_size = 1
num_epochs = 1


dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
df = dataset['train'].to_pandas()

df = df[["text", "hate_speech_score"]].dropna()

train_df, temp_df = train_test_split(df, test_size=train_size)
valid_df, test_df = train_test_split(temp_df, test_size=test_size)

train_text = train_df["text"].to_list()
train_labels = train_df["hate_speech_score"].to_list()

val_text = valid_df["text"].to_list()
val_labels = valid_df["hate_speech_score"].to_list()

test_text = test_df["text"].to_list()
test_labels = test_df["hate_speech_score"].to_list()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-small")
tokenizer.pad_token = tokenizer.eos_token


metric = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}


# Create the different tokenized dataset and prepare the dataloader
train_dataset = HateSpeechDataset(train_text, train_labels, tokenizer)
val_dataset = HateSpeechDataset(val_text, val_labels, tokenizer)
test_dataset = HateSpeechDataset(test_text, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
torch.cuda.empty_cache()

# Prepare the model using the pretrained GPT2 config and model
config = GPT2Config.from_pretrained("gpt2-small")
model = GPT2Regression.from_pretrained("gpt2-small", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Similar to the code from Practice 5 notebook
lr_scheduler = get_scheduler("linear", 
                             optimizer=optimizer, 
                             num_warmup_steps=5, 
                             num_training_steps=len(train_dataset) * 3)

all_epoch_results = []
loss_at_epoch = []

regression_metric = dict(zip(metric.keys(), torch.zeros(len(metric), device=device)))

print(f"Training on {device}")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    epoch_predictions = []
    epoch_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

        epoch_result = {
            "epoch": epoch + 1,
        }
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()
        
        total_loss += loss.item()

        epoch_result["loss"] = loss.item()
        
        with torch.no_grad():
            pred = outputs["logits"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            epoch_predictions.extend(pred.tolist())
            epoch_labels.extend(labels.tolist())

    for metric_name, metric_fn in metric.items():
        epoch_result[metric_name] = metric_fn(epoch_predictions, epoch_labels)
        

    all_epoch_results.append(epoch_result)
    
    epoch_result["epoch_loss"] = total_loss / len(train_loader)
    all_epoch_results.append(epoch_result)

    pd.DataFrame(all_epoch_results).to_csv("epoch_results.csv", index=False)

    loss_at_epoch.append(total_loss/len(train_loader))


model.save_weights("./gpt2-small-test-regression-finetuned-fixed")
tokenizer.save_pretrained("./gpt2-small-test-regression-finetuned-fixed")
dataframe = pd.DataFrame(loss_at_epoch, columns=["loss"])
dataframe.to_csv("small_test.csv", index=False)

run_metric_evalution("./gpt2-small-test-regression-finetuned-fixed",
                     test_loader=val_loader,
                     metric=metric,
                     num_epoch=5,
                     log_path = "./eval-logs-gpt2-small-test/",
                     device=device)
