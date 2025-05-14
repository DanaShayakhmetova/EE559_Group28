########################################################################
import torch
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from gpt2_regression_model import GPT2Regression
import gradio as gr # NEEDS TO BE 4.31.0
import re
import traceback

# For FastAPI, Pydantic, and Uvicorn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List 
########################################################################

# Loading models (same as before)
bert_tokenizer_global = None
id2label_global = {0: "Explicit Hate", 1: "Implicit Hate", 2: "Non-Hate"}
device_global = None

def load_all_models():
    global bert_tokenizer_global, bert_model_global, rasch_tokenizer_global, rasch_model_global
    global implicit_tokenizer_global, implicit_model_global, device_global
    print("Loading all models...")
    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_global}")

    #BERT
    bert_model_path = "saved_model_stg1_bert"
    try:
        print(f"Loading BERT tokenizer from: {bert_model_path}")
        bert_tokenizer_global = BertTokenizerFast.from_pretrained(bert_model_path)
        print(f"Loading BERT model from: {bert_model_path}")
        bert_model_global = BertForSequenceClassification.from_pretrained(bert_model_path)
        bert_model_global.to(device_global)
        bert_model_global.eval()
        print("BERT Classifier loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load BERT model from '{bert_model_path}': {e}")

    # RASCH
    rasch_model_identifier = "gpt2"
    try:
        print(f"Loading Rasch tokenizer from: {rasch_model_identifier}")
        rasch_tokenizer_global = GPT2Tokenizer.from_pretrained(rasch_model_identifier)
        if rasch_tokenizer_global.pad_token is None:
            rasch_tokenizer_global.pad_token = rasch_tokenizer_global.eos_token
        print(f"Loading Rasch model (GPT2Regression) from: {rasch_model_identifier}")
        rasch_model_global = GPT2Regression.from_pretrained(rasch_model_identifier)
        rasch_model_global.to(device_global)
        rasch_model_global.eval()
        print("GPT2 Regression (Rasch) loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Rasch model from '{rasch_model_identifier}': {e}")

    # IMPLICIT GPT
    implicit_model_path = "gpt2-implicit-hate/final"
    try:
        print(f"Loading Implicit Explanation tokenizer from: {implicit_model_path}")
        implicit_tokenizer_global = GPT2Tokenizer.from_pretrained(implicit_model_path)
        if implicit_tokenizer_global.pad_token is None:
            implicit_tokenizer_global.pad_token = implicit_tokenizer_global.eos_token
        print(f"Loading Implicit Explanation model from: {implicit_model_path}")
        implicit_model_global = GPT2LMHeadModel.from_pretrained(implicit_model_path)
        implicit_model_global.to(device_global)
        implicit_model_global.eval()
        print("GPT2 Explanation Model loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Implicit Explanation model from '{implicit_model_path}': {e}")
    print("All models loaded successfully")
########################################################################

# HELPERS 
def classify_tweet_internal(text):
    inputs = bert_tokenizer_global(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device_global)
    with torch.no_grad():
        logits = bert_model_global(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        label_idx = torch.argmax(probs, dim=1).item()
    return id2label_global[label_idx]


def get_rasch_score_internal(text):
    inputs = rasch_tokenizer_global(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device_global)
    with torch.no_grad():
        output = rasch_model_global(**inputs)
        if isinstance(output, tuple):
            logits_tensor = output[0]
        elif hasattr(output, 'logits'):
            logits_tensor = output.logits
        elif isinstance(output, dict) and 'logits' in output:
             logits_tensor = output['logits']
        else:
            logits_tensor = output
        score = logits_tensor.squeeze().item()
    return round(score, 2)

def get_implicit_explanation_internal(text, max_length=50):
    prompt = f"Tweet: {text}\nReason it contains implicit hate:"
    inputs = implicit_tokenizer_global(prompt, return_tensors="pt", truncation=True, max_length=1024-max_length).to(device_global)
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        generate_kwargs = {
            "attention_mask": attention_mask, "max_new_tokens": max_length, "do_sample": True,
            "top_k": 50, "top_p": 0.95, "num_return_sequences": 1,
            "pad_token_id": implicit_tokenizer_global.eos_token_id,
        }
        if attention_mask is None: del generate_kwargs["attention_mask"]
        outputs_gen = implicit_model_global.generate(input_ids, **generate_kwargs)
    full_decoded_output = implicit_tokenizer_global.decode(outputs_gen[0], skip_special_tokens=True)
    explanation_parts = full_decoded_output.split("Reason it contains implicit hate:")
    if len(explanation_parts) > 1: raw_explanation = explanation_parts[-1].strip()
    else:
        raw_explanation = full_decoded_output.replace(prompt.split("\nReason")[0].strip(), "").strip()
        if raw_explanation.startswith("Reason it contains implicit hate:"):
            raw_explanation = raw_explanation.split("Reason it contains implicit hate:")[-1].strip()
    return raw_explanation

########################################################################

# GRADIO

def perform_tweet_analysis(tweet_text: str): # Renamed for clarity
    if not tweet_text or not tweet_text.strip():
        return "Please enter a tweet.", "N/A", "N/A"
    try:
        classification = classify_tweet_internal(tweet_text)
        rasch_score_display = "N/A"
        explanation_display = "N/A"
        if classification != "Non-Hate":
            rasch_score = get_rasch_score_internal(tweet_text)
            rasch_score_display = f"{rasch_score}"
            if classification == "Implicit Hate":
                raw_explanation = get_implicit_explanation_internal(tweet_text)
                target_group, implied_statement = "an unspecified group", "an unspecified harmful implication"
                target_group_match = re.search(r"Target Group:\s*(.*?)(?=\s*\||$)", raw_explanation, re.IGNORECASE)
                if target_group_match: target_group = target_group_match.group(1).strip()
                else: print(f"Warning: Could not parse 'Target Group' from: {raw_explanation}")
                # implied_statement_match = re.search(r"Implied Statement:\s*(.*?)(?=\s*\.\s*Implied Statement:|Nsبن|Implying that|Because it suggests|This implies|By stating|Suggesting that|$)", raw_explanation, re.IGNORECASE | re.DOTALL)
                implied_statement_match = re.search(
                    r"Implied Statement:\s*(.*?)"  # Capture the statement
                    r"(?="  # Start positive lookahead (what to stop BEFORE)
                        r"\s*\.\s*Implied Statement:"  # 1. A period then another "Implied Statement" label
                        r"|\s*\.(?!\s*Implied Statement:)"  # 2. A period NOT followed by "Implied Statement:" (i.e., a sentence-ending period)
                                                    #    The (?!\s*Implied Statement) is a negative lookahead.
                                                    #    This ensures we stop at a simple period, but allow the more specific
                                                    #    ". Implied Statement:" to take precedence if it matches.
                        r"|Nsبن"                        # 3. Specific noise/delimiters
                        r"|Implying that"
                        r"|Because it suggests"
                        r"|This implies"
                        r"|By stating"
                        r"|Suggesting that"
                        r"|$"                           # 4. End of string
                    r")",
                    raw_explanation,
                    re.IGNORECASE | re.DOTALL
                )
                if implied_statement_match:
                    implied_statement = implied_statement_match.group(1).strip()
                    implied_statement = re.sub(r"^(that|the idea that|the notion that)\s*", "", implied_statement, flags=re.IGNORECASE).strip()
                    if implied_statement.endswith('.'): implied_statement = implied_statement[:-1]
                else: print(f"Warning: Could not parse 'Implied Statement' from: {raw_explanation}")
                explanation_display = f"This tweet seems to be implicitly hateful because it is targeting {target_group.lower()} by implying that {implied_statement.lower()}."
            elif classification == "Explicit Hate":
                explanation_display = "This tweet contains explicit hate."
        return classification, str(rasch_score_display), explanation_display
    except Exception as e:
        print(f"Error during analysis function: perform_tweet_analysis")
        print(f"Error type: {type(e).__name__}, Error message: {e}")
        traceback.print_exc()
        return f"Error: {type(e).__name__}", "Error during analysis", "Please check terminal."

########################################################################

try:
    load_all_models()
except RuntimeError as e:
    print(f"FATAL: Model loading failed. FastAPI app will not launch. Error: {e}")
    exit(1)

########################################################################
# FastAPI APP SETUP

class ApiInput(BaseModel):
    data: List[str] 

app_fastapi = FastAPI()

# CORS middleware
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CUSTOM API !
@app_fastapi.post("/analyze_tweet_extension_api")
async def analyze_tweet_api_endpoint(payload: ApiInput):
    if not payload.data or not isinstance(payload.data, list) or len(payload.data) == 0:
        raise HTTPException(status_code=400, detail="Input data list is missing or empty.")
    
    tweet_text = payload.data[0]

    try:
        classification, rasch_score, explanation = perform_tweet_analysis(tweet_text)
        return {"data": [classification, rasch_score, explanation]}
    except Exception as e:
        print(f"Error in FastAPI endpoint /analyze_tweet_extension_api: {e}")
        traceback.print_exc()
        error_type = type(e).__name__
        raise HTTPException(status_code=500, detail={"data": [f"Server Error: {error_type}", "Analysis Failed", "Please check server logs."]})


########################################################################

# our UI

ui_tweet_input = gr.Textbox(lines=5, label="Enter a tweet (for UI testing):", placeholder="Type your tweet here...")
ui_classification_output = gr.Textbox(label="Classification (UI)")
ui_rasch_output = gr.Textbox(label="Rasch Hate Score (UI)")
ui_explanation_output = gr.Textbox(label="Implicit Hate Explanation (UI)")

gradio_ui_only = gr.Interface(
    fn=perform_tweet_analysis, 
    inputs=ui_tweet_input,
    outputs=[ui_classification_output, ui_rasch_output, ui_explanation_output],
    title="Hate Speech Analyzer (UI Test Page)",
    description="UI for manual testing. The Chrome extension uses a separate API endpoint."
)
app_fastapi = gr.mount_gradio_app(app_fastapi, gradio_ui_only, path="/ui_test")

########################################################################
# Main execution block
if __name__ == "__main__":
    print(f"Gradio version (for UI): {gr.__version__}")
    print("Launching FastAPI app with custom endpoint and Gradio UI for testing...")
    uvicorn.run(app_fastapi, host="0.0.0.0", port=7860)