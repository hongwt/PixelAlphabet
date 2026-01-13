"""
Tests for loss functions
"""
import pytest
import torch
from src.loss import (
    FocalLoss, 
    LabelSmoothingCrossEntropy,
    CombinedLoss,
    create_loss_function
)


def test_focal_loss():
    """Test FocalLoss."""
    loss_fn = FocalLoss(gamma=2.0, alpha=1.0)
    
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    
    loss = loss_fn(pred, target)
    
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_label_smoothing():
    """Test LabelSmoothingCrossEntropy."""
    loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
    
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    
    loss = loss_fn(pred, target)
    
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_combined_loss():
    """Test CombinedLoss with different configurations."""
    # With focal loss
    loss_fn = CombinedLoss(
        use_focal=True,
        use_label_smoothing=True,
        lambda_focal=0.5
    )
    
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    
    loss = loss_fn(pred, target)
    
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"
    
    # Without focal loss
    loss_fn_no_focal = CombinedLoss(
        use_focal=False,
        use_label_smoothing=True
    )
    
    loss_no_focal = loss_fn_no_focal(pred, target)
    assert loss_no_focal.item() > 0


def test_loss_gradients():
    """Test that gradients flow through loss functions."""
    pred = torch.randn(4, 36, requires_grad=True)
    target = torch.randint(0, 36, (4,))
    
    # Test each loss function
    loss_functions = [
        FocalLoss(),
        LabelSmoothingCrossEntropy(),
        CombinedLoss()
    ]
    
    for loss_fn in loss_functions:
        loss = loss_fn(pred, target)
        loss.backward(retain_graph=True)
        
        assert pred.grad is not None, f"No gradient for {loss_fn.__class__.__name__}"
        assert torch.isfinite(pred.grad).all(), f"Invalid gradient for {loss_fn.__class__.__name__}"
        
        # Clear gradients for next test
        pred.grad.zero_()


def test_loss_factory():
    """Test loss function factory."""
    # Test all loss types
    loss_types = ['ce', 'focal', 'label_smoothing', 'combined']
    
    for loss_type in loss_types:
        loss_fn = create_loss_function(loss_type)
        
        pred = torch.randn(4, 36)
        target = torch.randint(0, 36, (4,))
        
        loss = loss_fn(pred, target)
        
        assert loss.item() > 0, f"Loss should be positive for {loss_type}"
        assert torch.isfinite(loss), f"Loss should be finite for {loss_type}"


def test_label_smoothing_effect():
    """Test that label smoothing produces different loss than hard labels."""
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    
    # Hard labels (CE)
    ce_loss = torch.nn.CrossEntropyLoss()
    hard_loss = ce_loss(pred, target)
    
    # Smoothed labels
    smooth_loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
    smooth_loss = smooth_loss_fn(pred, target)
    
    # Losses should be different (but close)
    assert not torch.allclose(hard_loss, smooth_loss, rtol=1e-3)
    print(f"CE Loss: {hard_loss.item():.4f}, Smoothed Loss: {smooth_loss.item():.4f}")


def test_focal_loss_focusing():
    """Test that focal loss down-weights easy examples."""
    # Create predictions with high confidence (easy example)
    easy_pred = torch.zeros(1, 36)
    easy_pred[0, 5] = 10.0  # Very confident prediction
    easy_target = torch.tensor([5])
    
    # Create predictions with low confidence (hard example)
    hard_pred = torch.zeros(1, 36)
    hard_pred[0, 5] = 0.1  # Low confidence
    hard_target = torch.tensor([5])
    
    focal_loss = FocalLoss(gamma=2.0)
    
    easy_loss = focal_loss(easy_pred, easy_target)
    hard_loss = focal_loss(hard_pred, hard_target)
    
    # Hard example should have higher loss
    assert hard_loss > easy_loss, "Focal loss should produce higher loss for hard examples"
    print(f"Easy loss: {easy_loss.item():.4f}, Hard loss: {hard_loss.item():.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
