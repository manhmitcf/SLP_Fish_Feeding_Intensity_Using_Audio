import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseModel

class FishSwinTransformer(BaseModel):
    """
    Swin Transformer Tiny (Swin-T) model adapted for Fish Feeding Intensity Classification.
    
    Inherits from BaseModel.
    Loads pre-trained weights from ImageNet.
    Adapts 1-channel input (Spectrogram) to 3-channel input required by Swin.
    """
    def __init__(self, num_classes=4, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (default: 4).
            pretrained (bool): Whether to load ImageNet weights (default: True).
        """
        super(FishSwinTransformer, self).__init__()
        
        # 1. Load Pre-trained Swin-T
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        self.swin = models.swin_t(weights=weights)
        
        # 2. Modify the Head
        # Swin-T's head is a Linear layer. We replace it to match our num_classes.
        # self.swin.head.in_features gives the input dimension of the last layer (usually 768 for Swin-T)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)
        
        print(f"Initialized FishSwinTransformer with {num_classes} classes. Pretrained: {pretrained}")

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape [Batch, 1, 224, 224]
            
        Returns:
            Tensor: Logits of shape [Batch, num_classes]
        """
        # 1. Input Adaptation: 1 Channel -> 3 Channels
        # Swin Transformer expects RGB images (3 channels).
        # We repeat the single channel spectrogram 3 times.
        # Input: [Batch, 1, H, W] -> Output: [Batch, 3, H, W]
        x = x.repeat(1, 3, 1, 1)
        
        # 2. Pass through Swin Transformer
        # The modified head will output [Batch, num_classes]
        x = self.swin(x)
        
        return x

if __name__ == "__main__":
    # Simple test to verify the model structure
    model = FishSwinTransformer(num_classes=4)
    print(f"Total Parameters: {model.count_parameters():,}")
    
    # Dummy input: Batch=2, Channel=1, H=224, W=224
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}") # Should be [2, 4]
