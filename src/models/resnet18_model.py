"""
ResNet18 Model Architecture
---------------------------
Defines ResNet18 model for water quality classification.
Uses pretrained ResNet18 and modifies final layer for 3 classes.

Architecture:
    - Backbone: ResNet18 (pretrained on ImageNet)
    - Modified FC layer: 512 -> 3 classes (clean, muddy, polluted)
    - Can freeze/unfreeze backbone layers

Usage:
    from src.models.resnet18_model import create_model, load_model
    
    # Create new model
    model = create_model(num_classes=3, pretrained=True)
    
    # Load saved model
    model = load_model('models/resnet18/resnet18_test_model.pth')
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from src.utils.logger import get_logger
# from src.utils.logger import setup_logger  # for testing

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# Setup logger
# logger = setup_logger(__name__, log_file=os.path.join(LOG_DIR, 'resnet18_model.log'))  # Uncomment this line for standalone testing of this file and comment the next line
logger = get_logger(__name__)

# Class names
CLASS_NAMES = ['clean', 'muddy', 'polluted']


class WaterQualityResNet18(nn.Module):
    """
    ResNet18 model for water quality classification
    
    Architecture:
        - Input: (batch_size, 3, 224, 224)
        - ResNet18 backbone: Extracts features
        - FC layer: 512 -> 3 classes
        - Output: (batch_size, 3)
    
    Args:
        num_classes (int): Number of output classes (default: 3)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze backbone layers (default: False)
    """
    
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super(WaterQualityResNet18, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
           self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Get number of features from last layer
        num_features = self.resnet.fc.in_features
        
        # Replace final fully connected layer
        # Original: 512 -> 1000 (ImageNet classes)
        # Modified: 512 -> 3 (water quality classes)
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Class logits (batch_size, num_classes)
        """
        return self.resnet(x)
    
    
    def freeze_backbone(self):
        """
        Freeze all layers except final FC layer
        
        Useful for:
        - Fine-tuning only last layer
        - Faster training
        - Less memory usage
        """
        logger.info("Freezing backbone layers...")
        
        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze final FC layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        logger.info("Backbone frozen. Only FC layer trainable.")
    
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full fine-tuning
        """
        logger.info("Unfreezing all layers...")
        
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        logger.info("All layers trainable.")
    
    
    def get_trainable_params(self):
        """
        Get count of trainable parameters
        
        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return trainable, total


def create_model(num_classes=3, pretrained=True, freeze_backbone=False, device='cpu'):
    """
    Create ResNet18 model for water quality classification
    
    Args:
        num_classes (int): Number of classes (default: 3)
        pretrained (bool): Use ImageNet weights (default: True)
        freeze_backbone (bool): Freeze backbone (default: False)
        device (str): Device to load model ('cpu' or 'cuda')
        
    Returns:
        WaterQualityResNet18: Model instance
        
    Example:
        model = create_model(num_classes=3, pretrained=True)
        model = model.to('cuda')
    """
    
    logger.info("="*60)
    logger.info("Creating ResNet18 Model")
    logger.info("="*60)
    
    # Create model
    model = WaterQualityResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Move to device
    model = model.to(device)
    logger.info(f"Model is using device: {device}")
    
    # Log model info
    trainable, total = model.get_trainable_params()
    
    logger.info(f"Model: ResNet18")
    logger.info(f"Pretrained: {pretrained}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Backbone frozen: {freeze_backbone}")
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Device: {device}")
    logger.info("="*60)
    
    return model


def load_model(checkpoint_path, num_classes=3, device='cpu', strict=True):
    """
    Load saved model from checkpoint
    
    Args:
        checkpoint_path (str): Path to saved model (.pth file)
        num_classes (int): Number of classes
        device (str): Device to load model
        strict (bool): Strict state dict loading
        
    Returns:
        WaterQualityResNet18: Loaded model
        
    Example:
        model = load_model('models/resnet18/resnet18_test_model.pth', device='cuda')
    """
    
    logger.info("="*60)
    logger.info("Loading Model from Checkpoint")
    logger.info("="*60)
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    # Create model
    model = WaterQualityResNet18(num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    logger.info(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Checkpoint loaded at {device}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info("Loaded model from checkpoint dict (key: 'model_state_dict')")
            
            # Log additional info if available
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                logger.info(f"Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
            if 'loss' in checkpoint:
                logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint, strict=strict)
            logger.info("Loaded model state dict directly")
    else:
        model.load_state_dict(checkpoint, strict=strict)
        logger.info("Loaded model state dict directly")
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info("="*60)
    
    return model


def save_model(model, save_path, epoch=None, accuracy=None, loss=None, additional_info=None):
    """
    Save model checkpoint with metadata
    
    Args:
        model: Model to save
        save_path (str): Path to save checkpoint
        epoch (int): Current epoch number
        accuracy (float): Model accuracy
        loss (float): Model loss
        additional_info (dict): Any additional info to save
        
    Example:
        save_model(
            model, 
            'models/resnet18/resnet18_test_model.pth',
            epoch=10,
            accuracy=95.5,
            loss=0.125
        )
    """
    
    logger.info(f"Saving model to: {save_path}")
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Preparing checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': model.num_classes,
        'pretrained': model.pretrained
    }
    
    # Adding optional metadata
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    if loss is not None:
        checkpoint['loss'] = loss
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Saving checkpoint
    torch.save(checkpoint, save_path)
    
    logger.info(f"Model saved successfully to: {save_path}")
    if epoch is not None:
        logger.info(f"Epoch: {epoch}")
    if accuracy is not None:
        logger.info(f"Accuracy: {accuracy:.2f}%")
    if loss is not None:
        logger.info(f"Loss: {loss:.4f}")


def model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print model summary and test forward pass for a single image input
    Like Testing with dummy input 
    
    Args:
        model: Model instance
        input_size (tuple): Input tensor size
        
    Example:
        model = create_model()
        model_summary(model)
    """
    
    logger.info("="*60)
    logger.info("Model Summary")
    logger.info("="*60)
    
    # Model architecture
    logger.info("Architecture:")
    logger.info(str(model)) # Print model architecture
    
    # Parameter counts
    trainable, total = model.get_trainable_params()
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Total parameters: {total:,}")
    
    # Test forward pass
    logger.info("Testing forward pass...")
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Expected output: ({input_size[0]}, {model.num_classes})")
        
        logger.info("Forward pass successful!")
        
    except Exception as e:
        logger.error(f"Forward pass failed: {str(e)}")
    
    logger.info("="*60)


# Testing
if __name__ == "__main__":
    """
    Test model creation and basic operations
    Run: python src/models/resnet18_model.py
    """
    
    logger.info("="*60)
    logger.info("Testing ResNet18 Model")
    logger.info("="*60)
    
    # Test 1: Create model without pretraining
    logger.info("1. Creating model without pretrained weights...")
    try:
        present_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(num_classes=3, pretrained=False, device=present_device)
        logger.info("Model created successfully!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 2: Model summary
    logger.info("2. Getting model summary for non pretrained model...")
    try:
        model_summary(model)
        logger.info("Summary generated!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 3: Forward pass
    logger.info("3. Testing forward pass for dummy input in non pretrained model...")
    try:
        dummy_input = torch.randn(4, 3, 224, 224).to(present_device)  # Batch of 4 images(loaded to GPU if available)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Output sample: {output[0]}")
        logger.info("Forward pass successful!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 4: Freeze/Unfreeze
    logger.info("4. Testing freeze/unfreeze...")
    try:
        trainable_before, _ = model.get_trainable_params()
        logger.info(f"Trainable params before freeze: {trainable_before:,}")
        
        model.freeze_backbone()
        trainable_frozen, _ = model.get_trainable_params()
        logger.info(f"Trainable params after freeze: {trainable_frozen:,}")
        
        model.unfreeze_backbone()
        trainable_after, _ = model.get_trainable_params()
        logger.info(f"Trainable params after unfreeze: {trainable_after:,}")
        
        logger.info("Freeze/unfreeze working!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Test 5: Save and load
    logger.info("5. Testing save and load...")
    try:
        # Save model
        test_path = os.path.join(PROJECT_ROOT, "models", "resnet18","resnet18_test_model.pth")
        save_model(model, test_path, epoch=1, accuracy=85.5, loss=0.345)
        logger.info(f"Model saved at {test_path}!")
        
        # Load model
        present_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loaded_model = load_model(test_path, num_classes=3, device=present_device)
        logger.info(f"Model loaded from {test_path} at {present_device}!")
        
        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
            logger.info("Test file cleaned up")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    logger.info("="*60)
    logger.info("Model tests complete!")
    logger.info("="*60)