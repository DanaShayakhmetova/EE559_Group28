import torch
from transformers import BertTokenizer
from multi_teacher_distillation.multi_head_student import MultiHeadStudent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Createing MultiHeadStudent model...")
model_path = "./small_alpha=0.5_bert-base-uncased_multi_head_model/alpha=0.5_weights/"  
model = MultiHeadStudent(pretrained_model_name='bert-base-uncased', num_classes=3)
print("Loading model weights from:", model_path, " to the model...")
model.load_weights(model_path)
model.to(device)
model.eval()

print("Model loaded successfully.")
print("Starting inference...")
while True:
    # Prepare input text
    text = input("Enter a sentence to classify (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    # Tokenize and prepare inputs
    inputs = model.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass to get both outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, from_batch='both')

        # Extract outputs
        classification_logits = outputs['logits_c']
        regression_score = outputs['logits_r']

        # Process results
        predicted_class = torch.argmax(classification_logits, dim=-1).item()
        regression_score_value = regression_score.item()

        # Print results
        print(f"Predicted class: {predicted_class + 1}, logits: {classification_logits.cpu().numpy()}")
        print(f"Regression (Rasch score): {regression_score_value}")
