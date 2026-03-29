from torchvision import models
import torch.nn as nn

class WaterQualityResNet18(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()

        if pretrained:
            self.model= models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc.parameters():
                param.requires_grad = True

        else: # fine tuning
            for name, param in self.model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)