# tests/test_model.py
import pytest
import torch
from PIL import Image
import io
import numpy as np
from pathlib import Path
import sys
import os

# Add backend directory to Python path
backend_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))
if backend_src not in sys.path:
    sys.path.append(backend_src)

from app.services.model_service import ModelService
from app.services.preprocessor import ImagePreprocessor


@pytest.fixture
def model_service():
    """Create model service instance"""
    return ModelService()


@pytest.fixture
def preprocessor():
    """Create preprocessor instance"""
    return ImagePreprocessor()


@pytest.fixture
def sample_image():
    """Create sample test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


def test_model_initialization(model_service):
    """Test model initialization"""
    assert model_service.model is not None
    assert model_service.preprocessor is not None
    assert isinstance(model_service.class_mapping, dict)


def test_model_device(model_service):
    """Test model device assignment"""
    assert model_service.device in [torch.device('cpu'),
                                    torch.device('cuda'),
                                    torch.device('mps')]
    # Compare device types instead of full device objects
    model_device = next(model_service.model.parameters()).device
    assert model_device.type == model_service.device.type


def test_preprocessor_transforms(preprocessor):
    """Test image preprocessing transforms"""
    # Create test image and save it to BytesIO
    test_image = Image.new('RGB', (300, 300), color='red')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Process image
    processed = preprocessor.preprocess(img_byte_arr)

    # Check output shape and type
    assert isinstance(processed, torch.Tensor)
    assert len(processed.shape) == 4  # Batch, Channels, Height, Width
    assert processed.shape[1] == 3  # RGB channels


def test_model_prediction(model_service, sample_image):
    """Test model prediction"""
    # Get prediction
    status, confidence = model_service.predict(sample_image)

    # Check prediction format
    assert status in ['open', 'closed']
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1


def test_preprocessor_top_portion(preprocessor):
    """Test top portion extraction"""
    # Create test image
    test_image = Image.new('RGB', (300, 300), color='red')

    # Extract top portion
    top_portion = preprocessor.extract_top_portion(test_image)

    # Check dimensions
    assert top_portion.size[1] == int(300 * 0.3)  # 30% of height
    assert top_portion.size[0] == 300  # Same width


def test_model_with_different_sizes(model_service):
    """Test model with different image sizes"""
    sizes = [(100, 100), (224, 224), (300, 300)]

    for size in sizes:
        # Create test image
        img = Image.new('RGB', size, color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Get prediction
        status, confidence = model_service.predict(img_byte_arr)

        # Check prediction
        assert status in ['open', 'closed']
        assert 0 <= confidence <= 1


def test_model_corrupted_image(model_service):
    """Test model behavior with corrupted image"""
    # Create corrupted image data
    corrupted_data = io.BytesIO(b'corrupted image data')

    # Check if appropriate exception is raised
    with pytest.raises(Exception):
        model_service.predict(corrupted_data)

def test_preprocessor_normalization(preprocessor):
    """Test image normalization values for standard ImageNet normalization"""
    # Create white image and save it to BytesIO
    white_img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = io.BytesIO()
    white_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Process image
    white_tensor = preprocessor.preprocess(img_byte_arr)

    # For white pixels (255, 255, 255), after ImageNet normalization:
    # R: (1 - 0.485) / 0.229 ≈ 2.2489
    # G: (1 - 0.456) / 0.224 ≈ 2.4286
    # B: (1 - 0.406) / 0.225 ≈ 2.6400

    # Check if values are within expected range for ImageNet normalization
    assert -3 <= torch.min(white_tensor).item() <= 3
    assert -3 <= torch.max(white_tensor).item() <= 3

    # Check if the maximum value is close to what we expect for white pixels
    assert abs(torch.max(white_tensor).item() - 2.64) < 0.1