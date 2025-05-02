from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


############################################################################################################
# Loading trained model and tokenizer 
model = GPT2LMHeadModel.from_pretrained("./gpt2-implicit-hate/final")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-implicit-hate/final")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


############################################################################################################
# LIME Setup
explainer = LimeTextExplainer(class_names=["Not Similar", "Similar"])
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)  # 384-dim fast model

############################################################################################################
max_length = 256

def generate_output(text):
    inputs = tokenizer(
        f"Tweet: {text}",
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=24,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

############################################################################################################
# Predict function for LIME that returns similarity-based scores
def lime_predict(texts):
    base_output = generate_output(texts[0])
    base_embedding = encoder.encode(base_output).reshape(1, -1)

    scores = []
    for i, text in enumerate(texts):
        gen_output = generate_output(text)
        gen_embedding = encoder.encode(gen_output).reshape(1, -1)
        sim_score = cosine_similarity(gen_embedding, base_embedding)[0][0]
        scores.append([1 - sim_score, sim_score])
        print(f"[{i+1}/{len(texts)}] Done: {sim_score:.4f}")
    
    return np.array(scores)

############################################################################################################
# Tweet to explain
tweet_input = ": at least 60 white south africans were murder by racist blacks in the past two months #fuckmandela"

# Run explanation (reduced num_samples for testing speed)
explanation = explainer.explain_instance(tweet_input, lime_predict, num_features=8, num_samples=50)
print("\nGenerated Reasoning:", generate_output(tweet_input))
print("\nLIME Explanation:")
for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:.4f}")

############################################################################################################
# Saving response for XAI 

import json
import re

# Tweet input
tweet_input = ": at least 60 white south africans were murder by racist blacks in the past two months #fuckmandela"

# Generate the full response
generated_response = generate_output(tweet_input)

# Extract parts using regex (or string manipulation)
# Assumes the format is "Tweet: ... Target Group: ... | Implied Statement: ..."
match = re.search(
    r"Tweet:\s*(.*?)\s*Target Group:\s*(.*?)\s*\|\s*Implied Statement:\s*(.*)",
    generated_response,
    re.IGNORECASE
)

if match:
    tweet_text = match.group(1).strip()
    target_group = match.group(2).strip()
    implied_statement = match.group(3).strip()
else:
    tweet_text = generated_response
    target_group = ""
    implied_statement = ""

# Prepare structured result
result_data = {
    "tweet_input": tweet_input,
    "generated": {
        "tweet": tweet_text,
        "target_group": target_group,
        "implied_statement": implied_statement
    },
    "lime_explanation": explanation.as_list()
}

# Save to JSON
with open("lime_result.json", "w") as f:
    json.dump(result_data, f, indent=2)

print("Saved structured explanation to lime_result.json")

