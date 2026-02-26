def test_arithmetic():
    """Simple smoke test."""
    assert 1 + 1 == 2

import torch
print(torch.cuda.is_available())