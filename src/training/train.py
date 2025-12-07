"""
Training Script
---------------
Main script to train the water quality classification model.

Usage:
    python src/training/train.py
    
    # With custom config
    python src/training/train.py --config configs/experiment1.yaml
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT) #Look in project root for imports

from src.models.resnet18_model import create_model, save_model
from src.data.dataset_loader import get_dataloaders
from src.training.trainer import Trainer
from src.utils.config_loader import load_config
from src.utils.logger import create_timestamped_log


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments

    Examples:
    #### Default
    python train.py
     args.config = 'configs/config.yaml'
     args.device = None
     args.resume = None

    #### Custom
    python train.py --config configs/exp1.yaml --device cpu
     args.config = 'configs/exp1.yaml'
     args.device = 'cpu'
     args.resume = None

    #### Resume
    python train.py --resume models/checkpoint.pth
     args.resume = 'models/checkpoint.pth'
    """
    parser = argparse.ArgumentParser(description='Train Water Quality Classification Model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    return parser.parse_args()


def setup_device(device=None):
    """
    Setup training device
    
    Args:
        device (str): Requested device or None for auto-detect
        
    Returns:
        str: Device to use ('cuda' or 'cpu')
    """
    if device is not None:
        selected_device = device
    else:
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {selected_device}")
    
    if selected_device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    return selected_device


def main():
    """
    Main training function
    """
    # Setup logger
    LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
    global logger
    logger = create_timestamped_log('training', LOG_DIR)

    # Parse arguments
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Water Quality Classification - Training")
    logger.info("="*60)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    logger.info("="*60)
    logger.info("Loading Configuration")
    logger.info("="*60)
    
    try:
        config = load_config(args.config)
        logger.info(f"Config loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        sys.exit(1)
    
    # Setup device
    logger.info("="*60)
    logger.info("Setting Up Device")
    logger.info("="*60)
    
    device = setup_device(args.device)
    
    # Create dataloaders
    logger.info("="*60)
    logger.info("Loading Dataset")
    logger.info("="*60)
    
    try:
        train_loader, test_loader = get_dataloaders(
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Create model
    logger.info("="*60)
    logger.info("Creating Model")
    logger.info("="*60)
    
    try:
        model = create_model(
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone'],
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        sys.exit(1)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info("="*60)
        logger.info("Resuming from Checkpoint")
        logger.info("="*60)
        
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Resumed from: {args.resume}")
            
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                logger.info(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            sys.exit(1)
    
    # Create trainer
    logger.info("="*60)
    logger.info("Initializing Trainer")
    logger.info("="*60)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=config,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {str(e)}")
        sys.exit(1)
    
    # Start training
    logger.info("="*60)
    logger.info("Starting Training Loop")
    logger.info("="*60)
    
    try:
        history = trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!!!")
        logger.info("Saving current model...")
        
        # Save interrupted model
        save_dir = config['paths']['save_dir']
        interrupted_path = os.path.join(save_dir, 'interrupted_model.pth')
        save_model(
            model,
            interrupted_path,
            epoch=len(history['train_loss']),
            accuracy=history['val_acc'][-1] if history['val_acc'] else 0,
            loss=history['val_loss'][-1] if history['val_loss'] else 0
        )
        logger.info(f"Model saved to: {interrupted_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Save final model
    logger.info("="*60)
    logger.info("Saving Final Model")
    logger.info("="*60)
    
    try:
        save_dir = config['paths']['save_dir']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_path = os.path.join(
            save_dir, 
            f'final_model_{timestamp}_acc{trainer.best_val_acc:.2f}.pth'
        )
        
        save_model(
            model,
            final_path,
            epoch=config['training']['epochs'],
            accuracy=trainer.best_val_acc,
            loss=history['val_loss'][-1],
            additional_info={'history': history}
        )
        
        logger.info(f"Final model saved to: {final_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {str(e)}")
    
    # Training summary
    logger.info("="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"Total Epochs: {config['training']['epochs']}")
    logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    logger.info(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    logger.info(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    logger.info(f"Best Epoch: {trainer.best_epoch}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    logger.info("------------------------Training completed successfully!--------------------------")


if __name__ == "__main__":
    main()