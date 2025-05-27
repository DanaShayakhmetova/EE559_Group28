# EE559_Group28

Link to the weights for the trained model gpt2-rasch-scale (git doesn't support files that are too big)
https://www.swisstransfer.com/d/a953c5bd-5503-4944-934e-e4ad579719ca


#### GPT Implicit Hate Model:
[Swiss transfer link](https://www.swisstransfer.com/d/3fe1c998-f216-4dfa-8759-ac2dbf9b2236)

#### Loading trained model and tokenizer 
model = GPT2LMHeadModel.from_pretrained("gpt2-implicit-hate/final")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-implicit-hate/final")
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#### [UPDATED] GPT Implicit Hate Model:

**Small**:[Swiss transfer link](https://www.swisstransfer.com/d/1dc8448b-50c7-4a66-882b-78b8e1b3f938)

**Large**: Will add link soon, it's loading 


#### Bert model:
[Swiss transfer link](https://www.swisstransfer.com/d/e9ae2ef0-f406-4d08-91a4-39344d89d5b8)

### DeBERTa modelL

[Swiss transfer link] (https://www.swisstransfer.com/d/ad4e9e6e-8ee6-446c-8f79-6bc7884b3dea)
