import torch


class DummyDataset(torch.utils.data.Dataset):
    """
    A dummy dataset that returns a constant tensor.

    This is a hack, we really should be collecting datasets using a policy and environment.
    But this lets PyTorch Lightning run, while we actually collect episodes in the trainer.
    """

    def __init__(self, size: int = 1000) -> None:
        self.size = size

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a constant tensor."""
        return torch.tensor([0])
