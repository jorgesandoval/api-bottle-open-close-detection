# tests/test_api.py
import warnings
import pytest
import io
import json
from PIL import Image
import numpy as np
from pathlib import Path

# Filter warnings before imports
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter('ignore')

# Now do the imports
import sys
import os

# Add backend directory to Python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, backend_dir)

from backend.src.app.main import create_app

@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app('testing')
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'


def test_classify_endpoint_success(client, sample_image):
    """Test successful image classification"""
    data = {
        'image': (sample_image, 'test.jpg')
    }
    response = client.post('/api/classify',
                           data=data,
                           content_type='multipart/form-data')

    assert response.status_code == 200
    assert 'Bottle_Status' in response.json
    assert 'Confidence' in response.json
    assert isinstance(response.json['Confidence'], float)
    assert response.json['Bottle_Status'] in ['open', 'closed']


def test_classify_no_image(client):
    """Test classification without image"""
    response = client.post('/api/classify')
    assert response.status_code == 400
    assert 'error' in response.json


def test_classify_invalid_image_format(client):
    """Test classification with invalid image format"""
    # Create a text file instead of an image
    data = {
        'image': (io.BytesIO(b'not an image'), 'test.txt')
    }
    response = client.post('/api/classify',
                           data=data,
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'error' in response.json


def test_classify_empty_image(client):
    """Test classification with empty image"""
    data = {
        'image': (io.BytesIO(), 'empty.jpg')
    }
    response = client.post('/api/classify',
                           data=data,
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'error' in response.json


def test_invalid_endpoint(client):
    """Test non-existent endpoint"""
    response = client.get('/invalid')
    assert response.status_code == 404


def test_method_not_allowed(client):
    """Test wrong HTTP method"""
    response = client.get('/api/classify')
    assert response.status_code == 405


def test_large_image(client):
    """Test with an image that exceeds size limit"""
    # Create a large image
    large_img = Image.new('RGB', (5000, 5000), color='red')
    img_byte_arr = io.BytesIO()
    large_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    data = {
        'image': (img_byte_arr, 'large.jpg')
    }
    response = client.post('/api/classify',
                           data=data,
                           content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'error' in response.json