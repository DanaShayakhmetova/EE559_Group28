import torch

class LatentHateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {
            key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        } | {
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)