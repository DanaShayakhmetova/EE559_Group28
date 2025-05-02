
############################################################################################################
# importing the necessary libraries 
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from evaluate import load


############################################################################################################
#Loading and preprocessing
# df = pd.read_csv("implicit-hate-corpus/implicit_hate_v1_stg3_posts.tsv", sep='\t')
url = "https://raw.githubusercontent.com/DanaShayakhmetova/gpt_data/refs/heads/main/implicit_hate_v1_stg3_posts.tsv"
df = pd.read_csv(url, sep='\t')
df = df.dropna()
df["text"] = df.apply(
    lambda row: f"Tweet: {row['post'].strip()} Target Group: {row['target'].strip()} | Implied Statement: {row['implied_statement'].strip()}",
    axis=1
)
# Train/test split
train_df, test_df = train_test_split(df[["text"]], test_size=0.1, random_state=42)
train_df = train_df.sample(n=1000, random_state=42)
test_df = test_df.sample(n=100, random_state=42)
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

############################################################################################################
#Tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a native pad token

max_length = 256

def tokenize(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize, remove_columns=["text"])

############################################################################################################
# Loading model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
############################################################################################################
# Prep training
training_args = TrainingArguments(
    output_dir="./gpt2-implicit-hate",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

############################################################################################################
# train
trainer.train()


# save for LIME and XAI 
trainer.save_model("./gpt2-implicit-hate/final") 
tokenizer.save_pretrained("./gpt2-implicit-hate/final") 

############################################################################################################
# Evaluating with diff decoding strategies 
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")


# Load the raw test inputs for evaluation
test_inputs = test_df["text"].tolist()

def evaluate_decoding(method="greedy"):
    preds = []
    refs = []
    for text in tqdm(test_inputs):
        prompt = text.split("Target Group:")[0].strip() 
        # print("PROMPT: ", prompt)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).input_ids.to(device)

        if method == "greedy":
            output_ids = model.generate(input_ids, max_new_tokens=24,no_repeat_ngram_size=3)
        elif method == "top-p":
            output_ids = model.generate(input_ids, max_new_tokens=24, do_sample=True, top_p=0.92, top_k=0,no_repeat_ngram_size=3)
        elif method == "beam":
            output_ids = model.generate(input_ids, max_new_tokens=24, num_beams=3, early_stopping=True, no_repeat_ngram_size=3,eos_token_id=tokenizer.eos_token_id)
        else:
            raise ValueError("Invalid method")

        # decoding the generated output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # print("RAW GEN: ",generated_text)
        if "Target Group:" in generated_text:
            generated_text = generated_text.split("Target Group:")[1].strip()
            generated_text = "Target Group: " + generated_text
            generated_text = generated_text.split(".")[0].strip()
            generated_text += "."
        else:
            parts = generated_text.split(".")
            if len(parts) > 1:
                generated_text = parts[1].strip()
            else:
                generated_text = parts[0].strip()


        # print("GEN: ",generated_text)
        preds.append(generated_text)

        if "Target Group:" in text:
            reference = text.split("Target Group:")[1].strip()
            if not reference.endswith("."):
                reference += "."
            reference = "Target Group: " + reference
            # print("REF: ",reference)
            refs.append(reference)
        else:
            refs.append("")  

    bleu_result = bleu.compute(predictions=preds, references=refs)
    rouge_result = rouge.compute(predictions=preds, references=refs)
    bertscore_result = bertscore.compute(predictions=preds, references=refs, lang="en")

    return bleu_result, rouge_result, bertscore_result

############################################################################################################
# Evaluating
print("\nEvaluating with Beam Search...")
bleu_beam, rouge_beam, bertscore_result = evaluate_decoding("beam")
print("BLEU:", bleu_beam)
print("ROUGE-L:", rouge_beam['rougeL'])
print("BERTScore (F1):", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))


print("\nEvaluating with Greedy Decoding...")
bleu_gdy, rouge_gdy, bertscore_result= evaluate_decoding("greedy")
print("BLEU:", bleu_gdy)
print("ROUGE-L:", rouge_gdy['rougeL'])
print("BERTScore (F1):", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))


print("\nEvaluating with Top-p Sampling...")
bleu_topp, rouge_topp, bertscore_result = evaluate_decoding("top-p")
print("BLEU:", bleu_topp)
print("ROUGE-L:", rouge_topp['rougeL'])
print("BERTScore (F1):", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))

############################################################################################################





