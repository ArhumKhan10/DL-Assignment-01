"""Baseline CNN models for affect recognition."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class BaselineCNN(nn.Module):
    """Baseline CNN for affect recognition with separate heads for classification and regression."""
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 8,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classification layer
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        
        # Expression classification head
        self.expression_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Valence regression head
        self.valence_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Arousal regression head
        self.arousal_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Dictionary with 'expression', 'valence', 'arousal' keys
        """
        # Extract features
        features = self.backbone(x)
        if len(features.shape) == 4:  # If backbone returns 4D tensor
            features = self.feature_processor(features)
        else:  # If backbone returns 2D tensor
            features = features
        
        # Predictions
        expression = self.expression_head(features)
        valence = self.valence_head(features).squeeze(-1)
        arousal = self.arousal_head(features).squeeze(-1)
        
        return {
            'expression': expression,
            'valence': valence,
            'arousal': arousal
        }
    
    def get_feature_extractor(self):
        """Get feature extractor (backbone + feature processor)."""
        return nn.Sequential(self.backbone, self.feature_processor)


def create_resnet_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> BaselineCNN:
    """Create ResNet-based model."""
    return BaselineCNN(
        backbone='resnet50',
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def create_efficientnet_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> BaselineCNN:
    """Create EfficientNet-based model."""
    return BaselineCNN(
        backbone='efficientnet_b0',
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def create_vgg_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> BaselineCNN:
    """Create VGG-based model."""
    return BaselineCNN(
        backbone='vgg16',
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def create_mobilenet_model(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> BaselineCNN:
    """Create MobileNet-based model."""
    return BaselineCNN(
        backbone='mobilenet_v2',
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(model: nn.Module, input_shape: Tuple[int, int, int, int] = (2, 3, 224, 224)):
    """Test model with dummy input."""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        output = model(dummy_input)
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shapes:")
        for key, value in output.items():
            print(f"  {key}: {value.shape}")
        print(f"Trainable parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    # Test different models
    models_to_test = [
        create_resnet_model(),
        create_efficientnet_model(),
        create_vgg_model(),
        create_mobilenet_model()
    ]
    
    for model in models_to_test:
        test_model(model)
        print("-" * 50)

