# src/app/services/model_service.py
import torch
import torch.nn as nn
from torchvision import models
import logging
from pathlib import Path
from torchvision.models import ResNet18_Weights
from app.services.preprocessor import ImagePreprocessor
from app.config import BaseConfig

logger = logging.getLogger(__name__)


class BottleClassifier(nn.Module):
    """Bottle classifier model based on ResNet18"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Use weights parameter instead of pretrained
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ModelService:
    def __init__(self):
        self.device = self._setup_device()
        self.model = self._load_model()
        self.preprocessor = ImagePreprocessor()
        self.class_mapping = {0: "open", 1: "closed"}
        logger.info(f"Model service initialized using device: {self.device}")

    def _setup_device(self):
        """Setup the inference device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        """Load the trained model"""
        try:
            # Initialize model
            model = BottleClassifier()

            # Load trained weights
            model_path = BaseConfig.MODEL_PATH
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Set model to evaluation mode
            model.to(self.device)
            model.eval()

            logger.info(f"Model loaded successfully from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_file):
        """
        Make a prediction for a bottle image

        Args:
            image_file: File object containing the image

        Returns:
            tuple: (status: str, confidence: float)

        Raises:
            Exception: If there's an error during prediction
        """
        try:
            # Preprocess image
            image_tensor = self.preprocessor.preprocess(image_file)
            image_tensor = image_tensor.to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities).item()

            # Get predicted class
            predicted_status = self.class_mapping[prediction.item()]

            return predicted_status, confidence

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise