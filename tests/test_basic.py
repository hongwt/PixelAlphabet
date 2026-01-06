def test_environment_setup():
    """Verify that important libraries can be imported."""
    import cv2
    import numpy as np
    import sys
    
    assert sys.version_info >= (3, 10), "Python version should be 3.10+"
    assert np.__version__ is not None
    # Just checking we can access the module
    assert cv2.__version__ is not None

def test_arithmetic():
    """Simple smoke test."""
    assert 1 + 1 == 2
