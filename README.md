## EE559_Group28

### Multi-Head Model for Hate Speech Recognition

This is the repository for the project of the course EE559 - Deep Learning at EPFL in 2025 for the group 28.

This repository contains all files used to create the results mentioned in the project report. 

Weights for the Multi-Head Model (bert model weights, GPT-2 Rasch scale model and Multi-Head Model trained weights)
[Swiss transfer link](https://www.swisstransfer.com/d/64d5427c-63ee-4bdd-ba31-fe343f1d409b)

To work properly the weights must be added to at the root of the project

then run : 
`python3 multi_head_inference.py` to enter inference mode with the Multi-Head trained weights

- To run in inference with GPT2-Rasch scale, run:
`python3 gpt2_hate_speech/inference.py`

- To train GPT2-Rasch scale, run:
`python3 train_gpt2_regression.py`

- To train Multi-Head model, run:
`python3 train_multi_head.py` 



### Project Structure

The directory structure of the main files for this project is the following :

```
├── bert_latent_hatred <-- Baselines models BERT and DeBERTa on the implicit_hate_corpus dataset
│ 
├── implicit-hate-corpus <-- Project data files from the paper "Latent Hatred: A Benchmark for Understanding Implicit Hate Speech" 
│ 
├── twitter_hate_extension_updated <-- All necessary files to run the chrome extension
│
├── gpt2_hate_speech/inference.py <--  run in inference with GPT2-Rasch scale
│
├── multi_head_inference.py <-- inference mode with the Multi-Head trained weights
│
├── train_gpt2_regression.py <-- To train GPT2-Rasch scale
│
├── train_multi_head.py <-- train Multi-Head model
│
├──gpt_latent_py_files <-- GPT Model for the implicit_hate_corpus dataset
```

#### GPT2 Regression for Rasch scale model:

[Swiss transfer link](https://www.swisstransfer.com/d/30e38139-bb15-45c7-b0be-cc02d44ba79a)


#### GPT Implicit Hate Model:

For twitter extension:

**Small**:[Swiss transfer link](https://www.swisstransfer.com/d/1dc8448b-50c7-4a66-882b-78b8e1b3f938)

For reference:

**Large**:[Swiss transfer link](https://www.swisstransfer.com/d/bc9c2a00-9ec9-4996-8b25-fa2e890ecbe7)


#### Bert model:
[Swiss transfer link](https://www.swisstransfer.com/d/e9ae2ef0-f406-4d08-91a4-39344d89d5b8)

### DeBERTa model

[Swiss transfer link](https://www.swisstransfer.com/d/ad4e9e6e-8ee6-446c-8f79-6bc7884b3dea)
