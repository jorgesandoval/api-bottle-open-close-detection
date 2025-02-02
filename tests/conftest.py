# tests/conftest.py
import os
import sys
import pytest
import warnings
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_dir))

@pytest.fixture(autouse=True)
def ignore_warnings():
    """Automatically ignore specific warnings for all tests"""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*'imghdr' is deprecated.*")
    warnings.filterwarnings("ignore", message=".*parameter 'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")