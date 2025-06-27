import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class ResNetClassifier(nn.Module):
    """ResNet-50 classifier for bird species classification"""
    
    def __init__(self, num_classes: int = 200, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification layer"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

def create_classifier(model_type: str = 'resnet50', num_classes: int = 200, 
                     pretrained: bool = True, **kwargs) -> nn.Module:
    """Create a classifier model"""
    
    if model_type == 'resnet50':
        return ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    model = create_classifier('resnet50', num_classes=200)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}") 