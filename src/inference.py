"""
Inference Module for PixelNet

Load trained model and perform character recognition on 24x24 images.
"""
from typing import Union, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

from src.model import create_model
from src.dataset import label_to_char
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter  # 用于进行可视化


class CharacterPredictor:
    """
    Wrapper class for character recognition inference.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Create model
        model = create_model(num_classes=36, dropout_rate=0.4)
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded model from {self.checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"  Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict character from image.
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or tensor)
            return_confidence: Whether to return confidence score
        
        Returns:
            Predicted character (and confidence if requested)
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        logits = self.model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        
        # Get prediction
        confidence, predicted_label = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_label = predicted_label.item()
        
        # Convert to character
        predicted_char = label_to_char(predicted_label)
        
        if return_confidence:
            return predicted_char, confidence
        else:
            return predicted_char
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: list,
        return_confidence: bool = True
    ) -> list:
        """
        Predict characters for a batch of images.
        
        Args:
            images: List of images (same formats as predict())
            return_confidence: Whether to return confidence scores
        
        Returns:
            List of predictions (characters or (character, confidence) tuples)
        """
        # Preprocess all images
        image_tensors = [self._preprocess_image(img) for img in images]
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Forward pass
        logits = self.model(batch_tensor)
        probabilities = F.softmax(logits, dim=1)
        
        # Get predictions
        confidences, predicted_labels = torch.max(probabilities, dim=1)
        
        results = []
        for conf, label in zip(confidences.cpu().numpy(), predicted_labels.cpu().numpy()):
            char = label_to_char(int(label))
            if return_confidence:
                results.append((char, float(conf)))
            else:
                results.append(char)
        
        return results
    
    @torch.no_grad()
    def visualize_layer_outputs(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        log_dir: str = "./logs/layer_visualization"
    ) -> str:
        """
        Visualize the output of each layer in the model.
        
        Args:
            image: Input image (same formats as predict())
            log_dir: Directory to save TensorBoard logs
        
        Returns:
            Path to the log directory
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        writer = SummaryWriter(log_dir)
        
        # Add original image
        writer.add_image('0_Input/original', image_tensor[0], 0)
        
        # Dictionary to store layer outputs
        activations = {}
        layer_names = []
        
        # Hook function to capture layer outputs
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        hooks = []
        
        # Conv1 + BN + ReLU
        hooks.append(self.model.conv1.register_forward_hook(get_activation('1_conv1')))
        hooks.append(self.model.bn1.register_forward_hook(get_activation('2_bn1')))
        hooks.append(self.model.relu.register_forward_hook(get_activation('3_relu1')))
        
        # ResBlock 1
        hooks.append(self.model.res_block1.register_forward_hook(get_activation('4_res_block1')))
        
        # ResBlock 2
        hooks.append(self.model.res_block2.register_forward_hook(get_activation('5_res_block2')))
        
        # Attention
        hooks.append(self.model.attention.register_forward_hook(get_activation('6_attention')))
        
        # Global Average Pooling
        hooks.append(self.model.global_avg_pool.register_forward_hook(get_activation('7_global_pool')))
        
        # Forward pass to trigger hooks
        _ = self.model(image_tensor)
        
        # Visualize each layer's output
        for name in sorted(activations.keys()):
            activation = activations[name]
            
            # Handle different tensor shapes
            if activation.dim() == 4:  # (B, C, H, W)
                batch_size, num_channels, height, width = activation.shape
                
                # Select a subset of channels to visualize (max 16)
                num_vis_channels = min(16, num_channels)
                
                # Create a grid of feature maps
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                fig.suptitle(f'{name} - Shape: {list(activation.shape)}', fontsize=16)
                
                for idx in range(num_vis_channels):
                    row = idx // 4
                    col = idx % 4
                    ax = axes[row, col]
                    
                    # Get the feature map
                    feature_map = activation[0, idx].cpu().numpy()
                    
                    # Display
                    im = ax.imshow(feature_map, cmap='viridis')
                    ax.set_title(f'Ch {idx}', fontsize=10)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Hide unused subplots
                for idx in range(num_vis_channels, 16):
                    row = idx // 4
                    col = idx % 4
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
                # Save to TensorBoard
                writer.add_figure(name, fig, 0)
                plt.close(fig)
                
                # Also add as images (show first 16 channels)
                grid_tensor = activation[0, :num_vis_channels].unsqueeze(1)  # (C, 1, H, W)
                # Normalize each channel independently for better visualization
                for i in range(grid_tensor.shape[0]):
                    channel = grid_tensor[i]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val - min_val > 0:
                        grid_tensor[i] = (channel - min_val) / (max_val - min_val)
                
                writer.add_images(f'{name}_channels', grid_tensor, 0)
                
            elif activation.dim() == 2:  # (B, Features) - FC layer output
                # Visualize as bar chart
                features = activation[0].cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.bar(range(len(features)), features)
                ax.set_title(f'{name} - Feature Vector (length: {len(features)})')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Activation Value')
                plt.tight_layout()
                
                writer.add_figure(name, fig, 0)
                plt.close(fig)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Add model graph
        writer.add_graph(self.model, image_tensor)
        
        writer.close()
        
        print(f"✓ Layer visualizations saved to: {log_dir}")
        print(f"  Run 'tensorboard --logdir {log_dir}' to view")
        
        return log_dir
    
    def _preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image to model input format.
        
        Args:
            image: Input in various formats
        
        Returns:
            Tensor of shape (3, 24, 24)
        """
        # Convert to PIL Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, torch.Tensor):
            # Already a tensor, just ensure correct format
            if image.dim() == 2:  # (H, W)
                image = image.unsqueeze(0).repeat(3, 1, 1)
            return image
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Ensure 24x24 size
        if image.size != (24, 24):
            print(f"Warning: Resizing image from {image.size} to (24, 24)")
            image = image.resize((24, 24), Image.BILINEAR)
        
        # Apply transform
        tensor = self.transform(image)
        
        return tensor


def predict(
    checkpoint_path: str,
    image_path: str,
    device: str = 'auto'
) -> Tuple[str, float]:
    """
    Convenience function for single prediction.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_path: Path to input image
        device: Device to use
    
    Returns:
        (predicted_character, confidence)
    """
    predictor = CharacterPredictor(checkpoint_path, device)
    return predictor.predict(image_path, return_confidence=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Character recognition inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize layer outputs in TensorBoard')
    parser.add_argument('--log-dir', type=str, default='./logs/layer_visualization',
                        help='Directory to save visualization logs')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CharacterPredictor(args.checkpoint, args.device)
    
    # Run prediction
    char, confidence = predictor.predict(args.image, return_confidence=True)
    print(f"\nPrediction: '{char}' (confidence: {confidence:.4f})")
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating layer visualizations...")
        log_dir = predictor.visualize_layer_outputs(args.image, args.log_dir)
        print(f"\n✓ Visualizations complete!")
        print(f"  To view: tensorboard --logdir {log_dir}")
