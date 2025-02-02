# src/app/services/preprocessor.py
import torch
from torchvision import transforms
from PIL import Image
import logging
from app.config import BaseConfig

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self):
        """Initialize the image preprocessor with the same transforms used in training"""
        self.transform = transforms.Compose([
            transforms.Resize(BaseConfig.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_top_portion(self, image):
        """Extract the top 30% of the image where the bottle cap is located"""
        width, height = image.size
        crop_height = int(height * 0.3)
        return image.crop((0, 0, width, crop_height))

    def preprocess(self, image_file):
        """
        Preprocess an image file for model inference

        Args:
            image_file: File object containing the image

        Returns:
            torch.Tensor: Preprocessed image tensor

        Raises:
            Exception: If there's an error during preprocessing
        """
        try:
            # Open and convert image to RGB
            image = Image.open(image_file).convert('RGB')

            # Extract top portion of the image
            image = self.extract_top_portion(image)

            # Apply transformations
            image_tensor = self.transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            logger.debug("Image preprocessed successfully")
            return image_tensor

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    @staticmethod
    def denormalize_image(image_tensor):
        """
        Convert a normalized tensor back to a PIL Image for visualization

        Args:
            image_tensor: Normalized image tensor

        Returns:
            PIL.Image: Denormalized image
        """
        try:
            # Remove batch dimension if present
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)

            # Denormalize
            denorm = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            denormalized = denorm(image_tensor)

            # Convert to PIL Image
            to_pil = transforms.ToPILImage()
            return to_pil(denormalized)

        except Exception as e:
            logger.error(f"Error denormalizing image: {str(e)}")
            raise