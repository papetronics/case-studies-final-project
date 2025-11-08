import torch
import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader

from .model import SupervisedDiceScorer, observation_to_tensor
from .greedy_scoring_dataset import GreedyScoringDataset

class SupervisedScorerTrainer(L.LightningModule):
    
    def __init__(
        self,
        hidden_size: int,
        learning_rate: float,
        batch_size: int,
        dataset_size: int,
        num_hidden: int,
        dropout_rate: float = 0.1,
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
        _, _, score_predictions = self.model(observation_to_tensor(observation).unsqueeze(0).to(self.device))
        return score_predictions
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        observations, max_scoring_targets, score_targets = batch
        
        # Move to device
        observations = observations.to(self.device)
        max_scoring_targets = max_scoring_targets.to(self.device)
        score_targets = score_targets.to(self.device)
        
        # Get predictions for the entire batch at once
        _, scoring_probs, score_predictions = self.model(observations)

        # print(score_targets[0].detach().cpu().numpy())
        
        # Use MSE loss for score prediction (regression)
        score_loss = F.mse_loss(score_predictions, score_targets)
        
        # Use KL divergence for soft targets (handles multiple valid max categories)
        #max_scoring_loss = F.kl_div(
        #    F.log_softmax(scoring_probs, dim=1),
        #    max_scoring_targets,
        #    reduction='batchmean'
        #)
        
        # Combined loss
        loss = score_loss # + max_scoring_loss
        
        # Calculate MAE for score predictions
        mae = F.l1_loss(score_predictions, score_targets)
        
        # Calculate accuracy by sampling from scoring head
        #scoring_dist = torch.distributions.Categorical(F.softmax(scoring_probs, dim=1))
        #sampled_categories = scoring_dist.sample()  # Shape: (batch_size,)
        
        # Check if sampled categories match any allowed max values (soft targets > 0)
        #valid_targets = (max_scoring_targets > 0).float()  # Shape: (batch_size, 13)
        
        #sampled_one_hot = F.one_hot(sampled_categories, num_classes=13).float()  # Shape: (batch_size, 13)
        #correct_samples = (sampled_one_hot * valid_targets).sum(dim=1)  # Shape: (batch_size,)
        #accuracy = correct_samples.mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/score_loss', score_loss, prog_bar=False)
        # self.log('train/max_scoring_loss', max_scoring_loss, prog_bar=False)
        self.log('train/mae', mae, prog_bar=True)
        #self.log('train/accuracy', accuracy, prog_bar=True)
        
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        observations, max_scoring_targets, score_targets = batch
        
        # Move to device
        observations = observations.to(self.device)
        max_scoring_targets = max_scoring_targets.to(self.device)
        score_targets = score_targets.to(self.device)
        
        # Get predictions (model is automatically in eval mode, dropout disabled)
        _, scoring_probs, score_predictions = self.model(observations)
        
        # Use MSE loss for score prediction (regression)
        score_loss = F.mse_loss(score_predictions, score_targets)
        
        # Use KL divergence for soft targets
        #max_scoring_loss = F.kl_div(
        #    F.log_softmax(scoring_probs, dim=1),
        #    max_scoring_targets,
        #    reduction='batchmean'
        #)
        
        # Combined loss
        loss = score_loss # + max_scoring_loss
        
        # Calculate MAE for score predictions
        mae = F.l1_loss(score_predictions, score_targets)
        
        # Calculate accuracy by sampling from scoring head
        #scoring_dist = torch.distributions.Categorical(F.softmax(scoring_probs, dim=1))
        #sampled_categories = scoring_dist.sample()  # Shape: (batch_size,)
        
        # Check if sampled categories match any allowed max values (soft targets > 0)
        #valid_targets = (max_scoring_targets > 0).float()  # Shape: (batch_size, 13)
        #sampled_one_hot = F.one_hot(sampled_categories, num_classes=13).float()  # Shape: (batch_size, 13)
        #correct_samples = (sampled_one_hot * valid_targets).sum(dim=1)  # Shape: (batch_size,)
        #accuracy = correct_samples.mean()
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/score_loss', score_loss, prog_bar=False)
        #self.log('val/max_scoring_loss', max_scoring_loss, prog_bar=False)
        self.log('val/mae', mae, prog_bar=True)
        #self.log('val/accuracy', accuracy, prog_bar=True)
        
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