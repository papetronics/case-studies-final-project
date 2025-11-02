import torch
import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader

from .model import SupervisedDiceScorer
from .greedy_scoring_dataset import GreedyScoringDataset

class SupervisedScorerTrainer(L.LightningModule):
    
    def __init__(
        self,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        dataset_size: int = 10000,
        num_hidden: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = SupervisedDiceScorer(hidden_size=hidden_size, num_hidden=num_hidden, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        
        # Create datasets
        self.train_dataset = GreedyScoringDataset(size=dataset_size)
        self.val_dataset = GreedyScoringDataset(size=500)  # Small eval set
        
    def forward(self, observation: Dict[str, Any]) -> torch.Tensor:
        _, scoring_probs = self.model(observation)
        return scoring_probs
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        observations, targets = batch
        
        # Move to device
        observations = observations.to(self.device)
        targets = targets.to(self.device)
        
        # Get predictions for the entire batch at once
        _, scoring_probs = self.model(observations)
        
        # Cross-entropy loss for classification (handles batched input automatically)
        loss = F.cross_entropy(scoring_probs, targets)
        
        # Calculate accuracy
        preds = torch.argmax(scoring_probs, dim=1)
        accuracy = (preds == targets).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', accuracy, prog_bar=True)
        
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        observations, targets = batch
        
        # Move to device
        observations = observations.to(self.device)
        targets = targets.to(self.device)
        
        # Get predictions (model is automatically in eval mode, dropout disabled)
        _, scoring_probs = self.model(observations)
        
        # Cross-entropy loss for classification
        loss = F.cross_entropy(scoring_probs, targets)
        
        # Calculate accuracy
        preds = torch.argmax(scoring_probs, dim=1)
        accuracy = (preds == targets).float().mean()
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Monitor loss (minimize)
            factor=0.7,          # Reduce LR by 30% (less aggressive)
            patience=3,          # Wait 3 epochs before reducing (better for short training)
            min_lr=1e-6          # Don't go below this LR
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",    # Monitor training loss 
                "frequency": 1            # Check every epoch
            }
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)