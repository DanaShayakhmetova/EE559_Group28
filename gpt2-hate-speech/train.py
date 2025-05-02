import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import evaluate
from torch.optim import AdamW
from transformers import GPT2Config
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel, get_scheduler

from gpt2_regression_model import GPT2Regression
from hate_speech_dataset import HateSpeechDataset


dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
df = dataset['train'].to_pandas()

df = df[["text", "hate_speech_score"]].dropna()

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_text = train_df["text"].to_list()
train_labels = train_df["hate_speech_score"].to_list()

val_text = valid_df["text"].to_list()
val_labels = valid_df["hate_speech_score"].to_list()

test_text = test_df["text"].to_list()
test_labels = test_df["hate_speech_score"].to_list()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token

# Create the different tokenized dataset and prepare the dataloader
train_dataset = HateSpeechDataset(train_text, train_labels, tokenizer)
val_dataset = HateSpeechDataset(val_text, val_labels, tokenizer)
test_dataset = HateSpeechDataset(test_text, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
torch.cuda.empty_cache()

# Prepare the model using the pretrained GPT2 config and model
config = GPT2Config.from_pretrained("gpt2-large")
model = GPT2Regression.from_pretrained("gpt2-large", config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Similar to the code from Practice 5 notebook
lr_scheduler = get_scheduler("linear", 
                             optimizer=optimizer, 
                             num_warmup_steps=5, 
                             num_training_steps=len(train_dataset) * 3)

loss_at_epoch = []
print(f"Training on {device}")
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    loss_at_epoch.append(total_loss/len(train_loader))
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")


model.save_pretrained("./gpt2-large-regression-finetuned")
tokenizer.save_pretrained("./gpt2-large-regression-finetuned")
dataframe = pd.DataFrame(loss_at_epoch, columns=["loss"])
dataframe.to_csv("loss_at_epoch_large.csv", index=False)


