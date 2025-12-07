"""
Trainer Module
--------------
Handles training and validation loops for the model.

Usage:
    from src.training.trainer import Trainer
    
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from src.utils.logger import get_logger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
LOG_FILE = os.path.join(LOG_DIR, "training.log")

# Setup logger
logger = get_logger('training')


class Trainer:
    """
    Trainer class for model training and validation
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config (dict): Configuration dictionary
        device (str): Device to train on ('cuda' or 'cpu')
        
    Example:
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()
    """
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        logger.info(f"Moving model to {device}")
        self.model = self.model.to(device)
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler (learning rate scheduler)
        self.scheduler = self._get_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        logger.info("="*60)
        logger.info("Trainer Initialized")
        logger.info("="*60)
        logger.info(f"Device: {device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info("="*60)
    
    
    def _get_optimizer(self):
        """
        Create optimizer based on config
        
        Returns:
            torch.optim.Optimizer: Optimizer instance
        """
        optimizer_name = self.config['training']['optimizer'].lower()
        weight_decay = self.config['training'].get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Optimizer using: {optimizer.__class__.__name__}")
        return optimizer
    
    
    def _get_scheduler(self):
        """
        Create learning rate scheduler based on config
        
        Returns:
            Scheduler or None
        """
        scheduler_name = self.config['training'].get('scheduler', None)
        
        if scheduler_name == 'step':
            step_size = self.config['training'].get('step_size', 10)
            gamma = self.config['training'].get('gamma', 0.1)
            
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            logger.info(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
            return scheduler
        
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
            logger.info("Scheduler: CosineAnnealingLR")
            return scheduler
        
        else:
            logger.info("No scheduler used")
            return None
    
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (avg_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]",leave=True,dynamic_ncols=True)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{accuracy:.2f}%'
            })
        
        # Calculate averages
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    
    def validate_epoch(self, epoch):
        """
        Validate for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (avg_loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                accuracy = 100 * correct / total
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        # Calculate averages
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    
    def save_checkpoint(self, epoch, accuracy, loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
            accuracy (float): Validation accuracy
            loss (float): Validation loss
            is_best (bool): Whether this is the best model so far
        """
        save_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{timestamp}_acc{accuracy:.2f}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(save_dir, f'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    
    def train(self):
        """
        Main training loop
        
        Returns:
            dict: Training history
        """
        logger.info("="*60)
        logger.info("Started Training")
        logger.info("="*60+"\n")
        
        for epoch in range(1, self.epochs + 1):
            logger.info("-" * 60)
            logger.info(f"Current Epoch {epoch}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Check if best model
            is_best = val_acc >= self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                logger.info(f"New best model! Accuracy: {val_acc:.2f}%")
            
            # Save checkpoint
            save_freq = self.config['logging'].get('save_frequency', 5)
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, val_loss, is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning Rate changed: {old_lr:.6f} to {new_lr:.6f}")
                else:
                    logger.info(f"Current Learning Rate: {new_lr:.6f}")
            logger.info(f"Epoch {epoch}/{self.epochs} Complete!")
            logger.info("-" * 60+"\n")
        
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        logger.info("="*60)
        
        return self.history