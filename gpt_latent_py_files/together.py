############################################################################################################
# Import necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from evaluate import load # bleu, rouge, bertscore
import os

############################################################################################################
# Configuration
MODEL_NAME = "gpt2-large"
OUTPUT_DIR_BASE = "./gpt2_implicit_hate_combined"
TRAIN_OUTPUT_DIR = f"{OUTPUT_DIR_BASE}/training_output"
FINAL_MODEL_DIR = f"{OUTPUT_DIR_BASE}/final_model"
PREDICTIONS_DIR = f"{OUTPUT_DIR_BASE}/predictions"

MAX_SEQ_LENGTH = 256  # Max length for tokenization (prompt + target + implied during training)
MAX_NEW_TOKENS_GENERATION = 60 # Max new tokens to generate (for target + [SEP] + implied + [END])
BATCH_SIZE = 2 # Adjusted for potentially larger model / memory constraints
NUM_EPOCHS = 3
RANDOM_STATE = 42
LEARNING_RATE = 5e-5 # Common starting point for fine-tuning GPT-2

# Create output directories if they don't exist
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(f"{TRAIN_OUTPUT_DIR}/logs", exist_ok=True)


############################################################################################################
# Load and preprocess raw data
# Using the GT-SALT dataset link as primary
url_primary = "https://raw.githubusercontent.com/GT-SALT/implicit-hate/main/implicit_hate_v1_stg3_posts.tsv"
url_fallback = "https://raw.githubusercontent.com/DanaShayakhmetova/gpt_data/refs/heads/main/implicit_hate_v1_stg3_posts.tsv"

try:
    df = pd.read_csv(url_primary, sep='\t')
    print(f"Successfully loaded data from primary URL: {url_primary}")
except Exception as e:
    print(f"Failed to load data from primary URL: {e}")
    print(f"Attempting fallback URL: {url_fallback}")
    try:
        df = pd.read_csv(url_fallback, sep='\t')
        print(f"Successfully loaded data from fallback URL: {url_fallback}")
    except Exception as e_fallback:
        print(f"Failed to load data from fallback URL: {e_fallback}")
        raise # Re-raise the exception if both fail

# Ensure required columns exist
required_columns = ['post', 'target', 'implied_statement']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset must contain the columns: {required_columns}. Found: {df.columns.tolist()}")

df = df.dropna(subset=required_columns)
df = df[required_columns] # Keep only necessary columns

# Train/test split on the raw columns
train_df, test_df = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)

# Convert pandas to HuggingFace Dataset objects
# train_dataset_raw = Dataset.from_pandas(train_df.reset_index(drop=True))
# test_dataset_raw = Dataset.from_pandas(test_df.reset_index(drop=True))

# print(f"Train dataset size: {len(train_dataset_raw)}")
# print(f"Test dataset size: {len(test_dataset_raw)}")

############################################################################################################
# Tokenizer setup with special tokens

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Define special tokens to separate parts of the input
STR_TOKEN = "[STR]"
SEP_TOKEN = "[SEP]"
END_TOKEN = "[END]"
special_tokens_dict = {
    "additional_special_tokens": [STR_TOKEN, SEP_TOKEN, END_TOKEN]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens.")

# GPT-2 has no pad token, use eos_token for padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Using pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Get IDs for special tokens, useful for generation and parsing
STR_TOKEN_ID = tokenizer.convert_tokens_to_ids(STR_TOKEN)
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
END_TOKEN_ID = tokenizer.convert_tokens_to_ids(END_TOKEN)

print(f"Special token IDs: STR={STR_TOKEN_ID}, SEP={SEP_TOKEN_ID}, END={END_TOKEN_ID}")

############################################################################################################
# Custom tokenization function to add special tokens explicitly for training

def tokenize_structured_for_training(example):
    tweet = str(example["post"]).strip()
    target = str(example["target"]).strip()
    implied = str(example["implied_statement"]).strip()
    
    # Build structured string with special tokens separating parts
    # Format: [STR] tweet_text [SEP] target_group_text [SEP] implied_statement_text [END]
    full_text = f"{STR_TOKEN} {tweet} {SEP_TOKEN} {target} {SEP_TOKEN} {implied} {END_TOKEN}"
    
    encoding = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_attention_mask=True
    )
    # For Causal LM, labels are usually the input_ids themselves.
    # The model internally shifts them for next-token prediction.
    encoding["labels"] = encoding["input_ids"].copy()

    return encoding

# Apply tokenization to datasets
# print("Tokenizing datasets for training...")
# train_dataset = train_dataset_raw.map(tokenize_structured_for_training, remove_columns=train_dataset_raw.column_names, num_proc=4) # Using num_proc for faster mapping
# test_dataset = test_dataset_raw.map(tokenize_structured_for_training, remove_columns=test_dataset_raw.column_names, num_proc=4)

# print(f"Sample tokenized input (decoded): {tokenizer.decode(train_dataset[0]['input_ids'])}")
# print(f"Sample labels (decoded): {tokenizer.decode([l if l != -100 else tokenizer.pad_token_id for l in train_dataset[0]['labels']])}")


############################################################################################################
# Load GPT-2 model and resize embeddings for new tokens

# print(f"Loading pre-trained model: {MODEL_NAME}")
# model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
# model.resize_token_embeddings(len(tokenizer))  # Important after adding tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# print(f"Model loaded on: {device}")

############################################################################################################
# Prepare Trainer and training arguments

# training_args = TrainingArguments(
#     output_dir=TRAIN_OUTPUT_DIR,
#     eval_strategy="epoch",        # Evaluate at the end of each epoch
#     save_strategy="epoch",        # Save at the end of each epoch
#     learning_rate=LEARNING_RATE,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     num_train_epochs=NUM_EPOCHS,
#     weight_decay=0.01,
#     logging_dir=f"{TRAIN_OUTPUT_DIR}/logs",
#     logging_steps=100,             # Log every 100 steps
#     save_total_limit=2,           # Keep only the best and the last model
#     load_best_model_at_end=True,  # Load the best model based on eval loss at the end of training
#     metric_for_best_model="loss", # Use eval loss to determine the best model
#     greater_is_better=False,      # Lower loss is better
#     remove_unused_columns=False,  # We handle columns in map
#     fp16=torch.cuda.is_available(), # Enable mixed precision if on GPU
#     report_to="tensorboard",      # Log to tensorboard
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

############################################################################################################
# Start training

# print("Starting training...")
# trainer.train()

# # Save model and tokenizer
# print(f"Saving final best model and tokenizer to {FINAL_MODEL_DIR}")
# trainer.save_model(FINAL_MODEL_DIR) 
# tokenizer.save_pretrained(FINAL_MODEL_DIR) 
# print("Training complete. Model saved.")

############################################################################################################

# Evaluation setup

print("Setting up evaluation metrics (BLEU, ROUGE)...")
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

# Use the original test_df for reference texts
test_tweets_for_eval = test_df["post"].tolist()
ground_truth_targets = test_df["target"].tolist()
ground_truth_implied = test_df["implied_statement"].tolist()

def evaluate_model_on_test_set(current_model, current_tokenizer, decoding_method="greedy"):
    print(f"\nEvaluating with {decoding_method} decoding...")
    
    pred_targets_list = []
    ref_targets_list = []
    pred_implied_list = []
    ref_implied_list = []

    output_file_path = os.path.join(PREDICTIONS_DIR, f"predictions_{decoding_method}.txt")
    
    # Open file in write mode at the beginning to clear previous content for this method
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Evaluation using {decoding_method} decoding:\n\n")
        f_out.write("Per-Tweet Predictions:\n")
        f_out.write("----------------------\n\n")
        
        for i, tweet_text in enumerate(tqdm(test_tweets_for_eval, desc=f"Generating - {decoding_method}")):
            ref_target = str(ground_truth_targets[i]).strip()
            ref_implied = str(ground_truth_implied[i]).strip()

            # Prepare prompt: "[STR] tweet_text [SEP]"
            # The model should then generate: "target_group [SEP] implied_statement [END]"
            prompt_text = f"{STR_TOKEN} {tweet_text.strip()} {SEP_TOKEN}"
            
            # Tokenize the prompt.
            input_ids = current_tokenizer(
                prompt_text, 
                return_tensors="pt", 
                truncation=True, 
                # Max length for prompt to leave space for generation:
                max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS_GENERATION, 
                padding=False # No padding needed for single inference
            ).input_ids.to(device)

            generation_params = {
                "max_new_tokens": MAX_NEW_TOKENS_GENERATION,
                "no_repeat_ngram_size": 3,
                "eos_token_id": END_TOKEN_ID, # Stop generation when [END] is produced
                "pad_token_id": current_tokenizer.pad_token_id # Suppress warnings
            }

            if decoding_method == "greedy":
                output_ids_tensor = current_model.generate(input_ids, **generation_params)
            elif decoding_method == "top-p": # As per paper
                output_ids_tensor = current_model.generate(input_ids, do_sample=True, top_p=0.92, top_k=0, **generation_params)
            elif decoding_method == "beam": # As per paper
                output_ids_tensor = current_model.generate(input_ids, num_beams=3, early_stopping=True, **generation_params)
            else:
                raise ValueError(f"Invalid decoding_method: {decoding_method}")

            # Get only the generated tokens (after the prompt)
            generated_token_ids = output_ids_tensor[0][input_ids.shape[-1]:]
            
            # Decode generated tokens, keeping special tokens for parsing
            decoded_generation = current_tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            
            # Clean up pad tokens and EOS if they appear spuriously before [END]
            decoded_generation = decoded_generation.replace(current_tokenizer.pad_token, "").strip()

            # Parse the generated text: "target_group [SEP] implied_statement [END]"
            pred_target_str = ""
            pred_implied_str = ""

            # Split by the first [SEP]
            parts_after_prompt = decoded_generation.split(SEP_TOKEN, 1)
            
            # The first part is the potential target
            potential_target_chunk = parts_after_prompt[0].strip()
            # Remove [END] if it's part of the target (e.g., model generates "target [END]")
            pred_target_str = potential_target_chunk.split(END_TOKEN)[0].strip()

            if len(parts_after_prompt) > 1:
                # [SEP] was found, the second part contains the implied statement
                potential_implied_chunk = parts_after_prompt[1].strip()
                # Remove [END] from the implied statement part
                pred_implied_str = potential_implied_chunk.split(END_TOKEN)[0].strip()
            
            # Store for metric calculation
            pred_targets_list.append(pred_target_str)
            ref_targets_list.append(ref_target)
            pred_implied_list.append(pred_implied_str)
            ref_implied_list.append(ref_implied)

            # Write to output file
            f_out.write(f"Tweet: {tweet_text}\n")
            f_out.write(f"  Ref Target:  {ref_target}\n")
            f_out.write(f"  Pred Target: {pred_target_str}\n")
            f_out.write(f"  Ref Implied: {ref_implied}\n")
            f_out.write(f"  Pred Implied:{pred_implied_str}\n\n")

    # ---- Calculate metrics for Target Group ----
    # The paper mentions BLEU* and ROUGE-L* (max vs avg score) if multiple references exist.
    # Here we have single references, so BLEU and ROUGE-L are direct.
    # Adding a period if not present, as seen in original example scripts (might be specific to how GT metrics were calculated)
    fmt_preds_target = [p + "." if not p.endswith(".") and p else p for p in pred_targets_list]
    fmt_refs_target = [r + "." if not r.endswith(".") and r else r for r in ref_targets_list]
    
    bleu_target = bleu.compute(predictions=fmt_preds_target, references=[[r] for r in fmt_refs_target]) # BLEU expects list of lists for references
    rouge_target = rouge.compute(predictions=fmt_preds_target, references=fmt_refs_target)
    bertscore_target_results = bertscore.compute(predictions=fmt_preds_target, references=fmt_refs_target, lang="en", device=device)
    
    exact_matches_target = sum(1 for p, r in zip(pred_targets_list, ref_targets_list) if p.lower() == r.lower())
    accuracy_target = exact_matches_target / len(ref_targets_list) if ref_targets_list else 0

    # ---- Calculate metrics for Implied Statement ----
    fmt_preds_implied = [p + "." if not p.endswith(".") and p else p for p in pred_implied_list]
    fmt_refs_implied = [r + "." if not r.endswith(".") and r else r for r in ref_implied_list]

    bleu_implied = bleu.compute(predictions=fmt_preds_implied, references=[[r] for r in fmt_refs_implied])
    rouge_implied = rouge.compute(predictions=fmt_preds_implied, references=fmt_refs_implied)
    bertscore_implied_results = bertscore.compute(predictions=fmt_preds_implied, references=fmt_refs_implied, lang="en", device=device)

    exact_matches_implied = sum(1 for p, r in zip(pred_implied_list, ref_implied_list) if p.lower() == r.lower())
    accuracy_implied = exact_matches_implied / len(ref_implied_list) if ref_implied_list else 0
    
    # --- CONSOLE OUTPUT OF METRICS ---
    print(f"\n--- Target Group Metrics ({decoding_method}) ---")
    print(f"BLEU: {bleu_target}")
    print(f"ROUGE-L: {rouge_target['rougeL']:.4f}")
    if bertscore_target_results and 'f1' in bertscore_target_results and bertscore_target_results['f1']:
        print(f"BERTScore (F1 avg): {sum(bertscore_target_results['f1']) / len(bertscore_target_results['f1']):.4f}")
    else:
        print("BERTScore (F1 avg): N/A (or not computed)")
    print(f"Exact Match Accuracy: {accuracy_target:.2%}")

    print(f"\n--- Implied Statement Metrics ({decoding_method}) ---")
    print(f"BLEU: {bleu_implied}")
    print(f"ROUGE-L: {rouge_implied['rougeL']:.4f}")
    if bertscore_implied_results and 'f1' in bertscore_implied_results and bertscore_implied_results['f1']:
        print(f"BERTScore (F1 avg): {sum(bertscore_implied_results['f1']) / len(bertscore_implied_results['f1']):.4f}")
    else:
        print("BERTScore (F1 avg): N/A (or not computed)")
    print(f"Exact Match Accuracy: {accuracy_implied:.2%}")
    

    # --- APPEND SUMMARY METRICS TO THE TXT FILE ---
    with open(output_file_path, "a", encoding="utf-8") as f_out: # Open in append mode
        f_out.write("\n\nSummary Metrics:\n")
        f_out.write("----------------\n")
        
        f_out.write(f"\n--- Target Group Metrics ({decoding_method}) ---\n")
        f_out.write(f"BLEU: {bleu_target}\n")
        f_out.write(f"ROUGE-L: {rouge_target['rougeL']:.4f}\n")
        if bertscore_target_results and 'f1' in bertscore_target_results and bertscore_target_results['f1']:
            f_out.write(f"BERTScore (F1 avg): {sum(bertscore_target_results['f1']) / len(bertscore_target_results['f1']):.4f}\n")
        else:
            f_out.write("BERTScore (F1 avg): N/A (or not computed)\n")
        f_out.write(f"Exact Match Accuracy: {accuracy_target:.2%}\n")

        f_out.write(f"\n--- Implied Statement Metrics ({decoding_method}) ---\n")
        f_out.write(f"BLEU: {bleu_implied}\n")
        f_out.write(f"ROUGE-L: {rouge_implied['rougeL']:.4f}\n")
        if bertscore_implied_results and 'f1' in bertscore_implied_results and bertscore_implied_results['f1']:
            f_out.write(f"BERTScore (F1 avg): {sum(bertscore_implied_results['f1']) / len(bertscore_implied_results['f1']):.4f}\n")
        else:
            f_out.write("BERTScore (F1 avg): N/A (or not computed)\n")
        f_out.write(f"Exact Match Accuracy: {accuracy_implied:.2%}\n")


############################################################################################################
# Run evaluation with different decoding strategies

# Load the best model saved by Trainer for evaluation
print(f"\nLoading best model from {FINAL_MODEL_DIR} for evaluation...")
eval_model = GPT2LMHeadModel.from_pretrained(FINAL_MODEL_DIR).to(device)
eval_tokenizer = GPT2Tokenizer.from_pretrained(FINAL_MODEL_DIR)

# Ensure pad token is set after loading
if eval_tokenizer.pad_token is None: 
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

# Re-fetch END_TOKEN_ID for the loaded tokenizer, just in case (should be same)
END_TOKEN_ID = eval_tokenizer.convert_tokens_to_ids(END_TOKEN) 

# Decoding strategies mentioned in the paper (Table 4)
# GPT-gdy (greedy), GPT-top-p (nucleus), GPT-beam (beam search)
evaluation_methods = ["greedy", "top-p", "beam"] 
for method in evaluation_methods:
    evaluate_model_on_test_set(eval_model, eval_tokenizer, method)

print("\nEvaluation complete.")
print(f"Predictions saved in directory: {PREDICTIONS_DIR}")
print(f"Trained model saved in directory: {FINAL_MODEL_DIR}")