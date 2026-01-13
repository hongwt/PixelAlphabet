"""
Tests for PixelNet model (Simplified)
"""
import pytest
import torch
from src.model import (
    PixelNet, 
    create_model, 
    SEBlock, 
    ResidualBlock
)


def test_se_block():
    """Test SEBlock (Squeeze-and-Excitation) module."""
    se = SEBlock(channels=256, reduction=16)
    x = torch.randn(2, 256, 12, 12)
    
    output = se(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert torch.isfinite(output).all(), "Output should not contain inf or nan"


def test_residual_block():
    """Test ResidualBlock."""
    # Without stride
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 8, 8)
    output = block(x)
    assert output.shape == (2, 64, 8, 8)
    
    # With stride and channel change
    block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 8, 8)
    output = block(x)
    assert output.shape == (2, 128, 4, 4)


def test_model_instantiation():
    """Test PixelNet instantiation."""
    model = create_model(num_classes=36, dropout_rate=0.3)
    assert isinstance(model, PixelNet)
    
    num_params = model.get_num_params()
    print(f"Model has {num_params:,} parameters")
    
    # Should be relatively small (~500K-1.5M parameters)
    # The optimized model (formerly LightPixelNet) is efficient
    assert 400_000 < num_params < 2_000_000, \
        f"Expected ~400K-2M params, got {num_params:,}"


def test_forward_pass():
    """Test forward pass with different batch sizes."""
    model = create_model()
    model.eval()
    
    # Single image
    x = torch.randn(1, 3, 24, 24)
    output = model(x)
    assert output.shape == (1, 36)
    
    # Batch of images
    x = torch.randn(8, 3, 24, 24)
    output = model(x)
    assert output.shape == (8, 36)


def test_output_range():
    """Test that model outputs valid logits."""
    model = create_model()
    model.eval()
    
    x = torch.randn(4, 3, 24, 24)
    
    with torch.no_grad():
        logits = model(x)
    
    # Logits should be finite
    assert torch.isfinite(logits).all(), "Logits contain inf or nan"
    
    # After softmax, should sum to 1
    probs = torch.softmax(logits, dim=1)
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_gradient_flow():
    """Test that gradients flow through the model."""
    model = create_model()
    model.train()
    
    x = torch.randn(2, 3, 24, 24, requires_grad=True)
    labels = torch.randint(0, 36, (2,))
    
    # Forward pass
    outputs = model(x)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    
    # Check model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"


def test_model_modes():
    """Test train/eval mode switching."""
    model = create_model()
    
    # Training mode
    model.train()
    x = torch.randn(2, 3, 24, 24)
    out_train = model(x)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    
    # Outputs should be different due to dropout
    assert not torch.allclose(out_train, out_eval), \
        "Train and eval outputs should differ (dropout effect)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_compatibility():
    """Test model works on CUDA."""
    device = torch.device('cuda')
    model = create_model().to(device)
    
    x = torch.randn(4, 3, 24, 24, device=device)
    output = model(x)
    
    assert output.device.type == 'cuda'
    assert output.shape == (4, 36)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
