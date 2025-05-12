import torch
from torch.utils.data import DataLoader, Dataset


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        super().__init__()
        self.encodings = tokenizer(texts, 
                                   truncation=True, 
                                   padding="max_length", 
                                   max_length=max_length,
                                   return_tensors="pt")
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
