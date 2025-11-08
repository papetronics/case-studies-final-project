import torch


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor([0])
