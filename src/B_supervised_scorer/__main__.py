#!/usr/bin/env python
import pytorch_lightning as L

from B_supervised_scorer.trainer import SupervisedScorerTrainer
from utilities.initialize import initialize, finish, ConfigParam

def main():
    # Define configuration schema
    config_params = [
        ConfigParam('epochs', int, 50, 'Number of training epochs'),
        ConfigParam('episodes_per_batch', int, 32, 'Episodes per training batch', display_name='Episodes per batch'),
        ConfigParam('learning_rate', float, 1e-3, 'Learning rate', display_name='Learning rate'),
        ConfigParam('hidden_size', int, 64, 'Hidden layer size', display_name='Hidden size'),
        ConfigParam('num_hidden', int, 1, 'Number of hidden layers', display_name='Num hidden layers'),
        ConfigParam('dataset_size', int, 10000, 'Dataset size', display_name='Dataset size'),
    ]
    
    # Initialize project with configuration
    wandb_run, config, logger = initialize(
        scenario_name='supervised_scorer',
        config_params=config_params,
        description='Yahtzee Supervised Scorer',
        logger_name='supervised-training'
    )
    
    # Extract config values for easy access
    epochs = config['epochs']
    episodes_per_batch = config['episodes_per_batch']
    learning_rate = config['learning_rate']
    hidden_size = config['hidden_size']
    num_hidden = config['num_hidden']
    dataset_size = config['dataset_size']
    
    # Create model
    model = SupervisedScorerTrainer(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        batch_size=episodes_per_batch,
        dataset_size=dataset_size,
        num_hidden=num_hidden
    )
        
    # Create trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator='auto',  # Will use GPU if available
        devices='auto',
    )
    
    # Train
    trainer.fit(model)
    
    finish(wandb_run)

if __name__ == "__main__":
    main()