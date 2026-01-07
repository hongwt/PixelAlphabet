"""
PixelNet Model Architecture

Custom CNN with residual connections and spatial attention for 24x24 character recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Learns to focus on discriminative spatial regions.
    Reference: CBAM (Convolutional Block Attention Module)
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,  # avg + max pooling across channels
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        # Compute channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        
        # Generate attention map
        attention_map = self.conv(concat)  # (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention
        return x * attention_map


class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutional layers.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class PixelNet(nn.Module):
    """
    Custom CNN for 24x24 character recognition.
    
    Architecture:
        - Input: (B, 3, 24, 24)
        - Block 1: Conv layers, reduces to (B, 32, 12, 12)
        - Block 2: Residual block, reduces to (B, 64, 6, 6)
        - Block 3: Residual block, keeps (B, 128, 6, 6)
        - Block 4: Residual block, (B, 256, 6, 6)
        - Block 5: Residual block, (B, 512, 6, 6)
        - Spatial Attention
        - Global Average Pooling
        - FC layers with dropout
        - Output: (B, 37)
    """
    
    def __init__(self, num_classes: int = 37, dropout_rate: float = 0.4):
        super(PixelNet, self).__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24 -> 12x12
        
        # Residual blocks with progressive downsampling
        self.res_block1 = ResidualBlock(32, 64, stride=2)   # 12x12 -> 6x6
        self.res_block2 = ResidualBlock(64, 128, stride=1)  # 6x6 -> 6x6 (Modified stride to preserve spatial info)
        self.res_block3 = ResidualBlock(128, 256, stride=1) # 6x6 -> 6x6
        self.res_block4 = ResidualBlock(256, 512, stride=1) # 6x6 -> 6x6
        
        # Spatial attention
        self.attention = SpatialAttention(kernel_size=3)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 3, 24, 24)
        
        Returns:
            Logits (B, num_classes)
        """
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool1(x)  # (B, 32, 12, 12)
        
        # Residual blocks
        x = self.res_block1(x)  # (B, 64, 6, 6)
        x = self.res_block2(x)  # (B, 128, 6, 6)
        x = self.res_block3(x)  # (B, 256, 6, 6)
        x = self.res_block4(x)  # (B, 512, 6, 6)
        
        # Apply spatial attention
        x = self.attention(x)  # (B, 512, 6, 6)
        
        # Global pooling
        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # FC layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = 37, dropout_rate: float = 0.4) -> PixelNet:
    """
    Factory function to create PixelNet model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
    
    Returns:
        PixelNet instance
    """
    model = PixelNet(num_classes=num_classes, dropout_rate=dropout_rate)
    print(f"Created PixelNet with {model.get_num_params():,} parameters")
    return model


if __name__ == '__main__':
    # Test model instantiation and forward pass
    model = create_model()
    dummy_input = torch.randn(4, 3, 24, 24)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 37), f"Expected (4, 37), got {output.shape}"
    print("✓ Model test passed")
