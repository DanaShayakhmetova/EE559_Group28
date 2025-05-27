import torch
from gpt2_regression_model import GPT2Regression

tokenizer, config, model =GPT2Regression.load_model("../model_weights/gpt2-small-regression-finetuned-fixed-smaller-train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
model.eval()

while True:
    with torch.no_grad():
        to_evaluate = input("Enter the text to evaluate (or 'CTRL-D' to quit): \n")
        inputs = tokenizer(to_evaluate, return_tensors="pt").to(device)
        outputs = model(**inputs)
        print(f"Predicted Rasch hate score: {outputs['logits'].item()}")
