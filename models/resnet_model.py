import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseModel

class FishResNet(BaseModel):
    """
    ResNet50 model adapted for Fish Feeding Intensity Classification.
    
    Inherits from BaseModel.
    Loads pre-trained weights from ImageNet.
    Adapts input layer for 1-channel Spectrogram.
    """
    def __init__(self, num_classes=4, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (default: 4).
            pretrained (bool): Whether to load ImageNet weights (default: True).
        """
        super(FishResNet, self).__init__()
        
        # 1. Load Pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # 2. Modify Input Layer (Conv1)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We change in_channels from 3 to 1.
        # To keep pretrained weights, we can sum the weights across the channel dimension 
        # or just re-initialize. Here we re-initialize for simplicity as it learns fast.
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # 3. Modify the Head (Fully Connected Layer)
        # ResNet's final layer is named 'fc'.
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        print(f"Initialized FishResNet50 with {num_classes} classes. Pretrained: {pretrained}")

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape [Batch, 1, 224, 224]
            
        Returns:
            Tensor: Logits of shape [Batch, num_classes]
        """
        # No need to repeat channels because we modified conv1 to accept 1 channel.
        return self.resnet(x)

if __name__ == "__main__":
    # Simple test
    model = FishResNet(num_classes=4)
    print(f"Total Parameters: {model.count_parameters():,}")
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
