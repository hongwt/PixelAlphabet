"""
Tests for loss functions
"""
import pytest
import torch
from src.loss import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    CombinedLoss,
    FocalLabelSmoothingLoss,
    ConfusionPairContrastiveLoss,
    create_loss_function,
    get_confusable_alpha_weights,
    CONFUSED_PAIRS,
    CONFUSED_PAIR_INDICES,
    CONFUSABLE_CHARS,
    _char_to_label,
)


# -----------------------------------------------------------------------
# Legacy loss tests (regression)
# -----------------------------------------------------------------------


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
    """Test CombinedLoss with default configuration."""
    loss_fn = CombinedLoss()

    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))

    loss = loss_fn(pred, target)

    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_loss_gradients():
    """Test that gradients flow through all loss functions."""
    pred = torch.randn(4, 36, requires_grad=True)
    target = torch.randint(0, 36, (4,))

    loss_functions = [
        FocalLoss(),
        LabelSmoothingCrossEntropy(),
        FocalLabelSmoothingLoss(),
        CombinedLoss(use_contrastive=False),
    ]

    for loss_fn in loss_functions:
        loss = loss_fn(pred, target)
        loss.backward(retain_graph=True)

        assert pred.grad is not None, f"No gradient for {loss_fn.__class__.__name__}"
        assert torch.isfinite(pred.grad).all(), f"Invalid gradient for {loss_fn.__class__.__name__}"

        pred.grad.zero_()


def test_loss_factory():
    """Test loss function factory – all types including new ones."""
    loss_types = ['ce', 'focal', 'label_smoothing', 'focal_label_smoothing', 'combined']

    for loss_type in loss_types:
        loss_fn = create_loss_function(loss_type)

        pred = torch.randn(4, 36)
        target = torch.randint(0, 36, (4,))

        loss = loss_fn(pred, target)

        assert loss.item() > 0, f"Loss should be positive for {loss_type}"
        assert torch.isfinite(loss), f"Loss should be finite for {loss_type}"


def test_loss_factory_backward_compat():
    """Regression: old-style create_loss_function('combined') still works."""
    loss_fn = create_loss_function('combined')
    pred = torch.randn(4, 36)
    target = torch.randint(0, 36, (4,))
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_label_smoothing_effect():
    """Test that label smoothing produces different loss than hard labels."""
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))

    ce_loss = torch.nn.CrossEntropyLoss()
    hard_loss = ce_loss(pred, target)

    smooth_loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
    smooth_loss = smooth_loss_fn(pred, target)

    # Label smoothing redistributes probability mass → loss values will differ
    assert abs(smooth_loss.item() - hard_loss.item()) > 1e-6, \
        "Smoothed loss should differ from hard CE"


def test_focal_loss_focusing():
    """Test that focal loss down-weights easy examples."""
    easy_pred = torch.zeros(1, 36)
    easy_pred[0, 5] = 10.0
    easy_target = torch.tensor([5])

    hard_pred = torch.zeros(1, 36)
    hard_pred[0, 5] = 0.1
    hard_target = torch.tensor([5])

    focal_loss = FocalLoss(gamma=2.0)

    easy_loss = focal_loss(easy_pred, easy_target)
    hard_loss = focal_loss(hard_pred, hard_target)

    assert hard_loss > easy_loss, "Focal loss should produce higher loss for hard examples"


# -----------------------------------------------------------------------
# FocalLabelSmoothingLoss tests
# -----------------------------------------------------------------------


def test_focal_label_smoothing_basic():
    """FocalLabelSmoothingLoss produces positive, finite loss."""
    loss_fn = FocalLabelSmoothingLoss(epsilon=0.1, gamma=3.0)
    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_focal_label_smoothing_hard_example_amplification():
    """Hard examples (low p_t) should receive amplified loss."""
    loss_fn = FocalLabelSmoothingLoss(epsilon=0.1, gamma=3.0,
                                       alpha=torch.ones(36))

    # Easy example: very confident correct prediction
    easy_pred = torch.zeros(1, 36)
    easy_pred[0, 5] = 10.0
    easy_target = torch.tensor([5])

    # Hard example: low-confidence prediction
    hard_pred = torch.zeros(1, 36)
    hard_pred[0, 5] = 0.1
    hard_target = torch.tensor([5])

    easy_loss = loss_fn(easy_pred, easy_target)
    hard_loss = loss_fn(hard_pred, hard_target)

    assert hard_loss.item() > easy_loss.item(), \
        "FocalLabelSmoothingLoss should amplify hard examples"


def test_focal_label_smoothing_class_alpha():
    """Confusable classes should receive higher alpha weight."""
    alpha = get_confusable_alpha_weights(36, base_weight=1.0, confusable_weight=2.0)
    loss_fn = FocalLabelSmoothingLoss(epsilon=0.1, gamma=3.0, alpha=alpha)

    # Same logits, but target differs: confusable vs non-confusable class
    pred = torch.randn(1, 36)

    # '8' is confusable (alpha=2.0), 'W' is not (alpha=1.0)
    idx_8 = _char_to_label('8')
    idx_W = _char_to_label('W')

    loss_8 = loss_fn(pred, torch.tensor([idx_8]))
    loss_W = loss_fn(pred, torch.tensor([idx_W]))

    # With same logits and higher alpha, confusable class loss should be higher
    assert loss_8.item() > loss_W.item(), \
        "Confusable class '8' (alpha=2.0) should have higher loss than 'W' (alpha=1.0)"


def test_focal_label_smoothing_gradient_flow():
    """Ensure gradients flow through FocalLabelSmoothingLoss."""
    pred = torch.randn(4, 36, requires_grad=True)
    target = torch.randint(0, 36, (4,))
    loss_fn = FocalLabelSmoothingLoss()
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_focal_label_smoothing_auto_alpha():
    """When alpha is None, auto-generates confusable weights."""
    loss_fn = FocalLabelSmoothingLoss(num_classes=36)
    # alpha buffer should exist and have correct shape
    assert hasattr(loss_fn, 'alpha')
    assert loss_fn.alpha.shape == (36,)
    # Check a confusable char has higher weight
    assert loss_fn.alpha[_char_to_label('8')] > loss_fn.alpha[_char_to_label('W')]


# -----------------------------------------------------------------------
# ConfusionPairContrastiveLoss tests
# -----------------------------------------------------------------------


def test_contrastive_loss_no_pairs_in_batch():
    """When no confusable pair is present, loss should be zero."""
    loss_fn = ConfusionPairContrastiveLoss(margin=0.5)

    # Only use class 35 (label 'Z') – no pair companion in batch
    features = torch.randn(4, 256)
    targets = torch.full((4,), 35, dtype=torch.long)  # all same class

    loss = loss_fn(features, targets)
    assert loss.item() == 0.0, "Loss should be 0 when no confusable pair is present"


def test_contrastive_loss_with_pair():
    """When a confusable pair is present, loss should be computable and non-negative."""
    loss_fn = ConfusionPairContrastiveLoss(margin=0.5)

    # Force pair (8, B) to appear
    idx_8 = _char_to_label('8')
    idx_B = _char_to_label('B')

    features = torch.randn(8, 256)
    targets = torch.tensor([idx_8, idx_8, idx_8, idx_8, idx_B, idx_B, idx_B, idx_B])

    loss = loss_fn(features, targets)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_contrastive_loss_identical_features_high_loss():
    """Identical features for both classes -> cosine dist ≈ 0 -> high loss."""
    loss_fn = ConfusionPairContrastiveLoss(margin=0.5)

    idx_8 = _char_to_label('8')
    idx_B = _char_to_label('B')

    # Identical features for both classes
    shared_feat = torch.randn(1, 256)
    features = shared_feat.expand(4, -1).clone()
    targets = torch.tensor([idx_8, idx_8, idx_B, idx_B])

    loss = loss_fn(features, targets)
    # Distance ~ 0, so loss ≈ margin
    assert loss.item() > 0.0, "Identical features should produce positive contrastive loss"


def test_contrastive_loss_orthogonal_features_zero_loss():
    """Orthogonal features -> cosine dist = 1.0 > margin -> zero loss."""
    loss_fn = ConfusionPairContrastiveLoss(
        confused_pairs=[(_char_to_label('8'), _char_to_label('B'))],
        margin=0.5,
    )

    idx_8 = _char_to_label('8')
    idx_B = _char_to_label('B')

    # Orthogonal features
    feat_8 = torch.zeros(1, 256)
    feat_8[0, 0] = 1.0
    feat_B = torch.zeros(1, 256)
    feat_B[0, 1] = 1.0

    features = torch.cat([feat_8, feat_B], dim=0)
    targets = torch.tensor([idx_8, idx_B])

    loss = loss_fn(features, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-5), \
        "Orthogonal features should produce zero contrastive loss"


def test_contrastive_loss_gradient_flow():
    """Ensure gradients flow through ConfusionPairContrastiveLoss."""
    loss_fn = ConfusionPairContrastiveLoss(margin=0.5)

    idx_8 = _char_to_label('8')
    idx_B = _char_to_label('B')

    features = torch.randn(4, 256, requires_grad=True)
    targets = torch.tensor([idx_8, idx_8, idx_B, idx_B])

    loss = loss_fn(features, targets)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


# -----------------------------------------------------------------------
# Updated CombinedLoss tests
# -----------------------------------------------------------------------


def test_combined_loss_with_contrastive():
    """CombinedLoss with contrastive component."""
    loss_fn = CombinedLoss(use_contrastive=True, lambda_contrastive=0.3)

    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))
    features = torch.randn(8, 256)

    loss = loss_fn(pred, target, features)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_combined_loss_without_contrastive():
    """CombinedLoss without contrastive component (features not needed)."""
    loss_fn = CombinedLoss(use_contrastive=False)

    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))

    # Should work without features argument
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_combined_loss_features_none_skips_contrastive():
    """When features=None is passed, contrastive component is skipped."""
    loss_fn = CombinedLoss(use_contrastive=True)

    pred = torch.randn(8, 36)
    target = torch.randint(0, 36, (8,))

    # features=None should not raise
    loss = loss_fn(pred, target, features=None)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_combined_loss_gradient_flow():
    """Gradients flow through CombinedLoss including contrastive."""
    pred = torch.randn(4, 36, requires_grad=True)
    features = torch.randn(4, 256, requires_grad=True)
    target = torch.randint(0, 36, (4,))

    loss_fn = CombinedLoss(use_contrastive=True)
    loss = loss_fn(pred, target, features)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# -----------------------------------------------------------------------
# Utility function tests
# -----------------------------------------------------------------------


def test_confusable_alpha_weights():
    """get_confusable_alpha_weights returns correct shape and values."""
    alpha = get_confusable_alpha_weights(36, base_weight=1.0, confusable_weight=2.0)
    assert alpha.shape == (36,)

    # Non-confusable class should be 1.0
    idx_W = _char_to_label('W')
    assert alpha[idx_W].item() == pytest.approx(1.0)

    # Confusable class should be 2.0
    idx_8 = _char_to_label('8')
    assert alpha[idx_8].item() == pytest.approx(2.0)


def test_confused_pairs_definitions():
    """Verify CONFUSED_PAIRS and CONFUSED_PAIR_INDICES are consistent."""
    assert len(CONFUSED_PAIRS) == 8
    assert len(CONFUSED_PAIR_INDICES) == 8

    for (char_a, char_b), (idx_a, idx_b) in zip(CONFUSED_PAIRS, CONFUSED_PAIR_INDICES):
        assert idx_a == _char_to_label(char_a)
        assert idx_b == _char_to_label(char_b)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
