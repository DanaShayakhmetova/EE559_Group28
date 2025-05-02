from transformers import pipeline
import torch 
import json


############################################################################################################
# Open JSON LIME
with open("lime_result.json", "r") as f:
    data = json.load(f)

# A check
# print("Tweet:", data["tweet_input"])
# print("Generated Tweet:", data["generated"]["tweet"])
# print("Target Group:", data["generated"]["target_group"])
# print("Implied Statement:", data["generated"]["implied_statement"])

# print("Explanation:")
# for feature, weight in data["lime_explanation"]:
#     print(f"{feature}: {weight:.4f}")

############################################################################################################
# XAI 
device = 0 if torch.cuda.is_available() else -1
# generator = pipeline("text-generation", model="gpt2-medium", device=device) 
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=device) 

# maybe once we put into cluset, we can try flan-t5-xl?

# models that work partially 
# google/flan-t5-large

tweet = data["tweet_input"]
target_group = data["generated"]["target_group"]
implied_statement = data["generated"]["implied_statement"]
lime_weights = {feature: weight for feature, weight in data["lime_explanation"]}
top_words_str = ", ".join([f"{word} ({weight:.4f})" for word, weight in sorted(lime_weights.items(), key=lambda x: -abs(x[1]))[:5]])

############################################################################################################
# # Prompt
prompt = f"""
Create a short text explanation with this information about implicit hate: 
Tweet: "{tweet}" 
Hates on: {target_group} by implying that "{implied_statement}".
Words that contribute to this implication are: {top_words_str}.
"""

############################################################################################################
# Run it
# response = generator(prompt, max_new_tokens=200)

# kinda works ?
response = generator(prompt, max_new_tokens=200, do_sample=True,temperature =0.7,top_p=0.9) 
print(response[0]["generated_text"])
