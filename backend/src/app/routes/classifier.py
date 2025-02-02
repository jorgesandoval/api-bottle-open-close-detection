# src/app/routes/classifier.py
from flask import Blueprint, request, jsonify
import logging
from app.services.model_service import ModelService
from app.utils.validators import validate_image
from app.config import BaseConfig

# Initialize blueprint and logger
classifier_bp = Blueprint('classifier', __name__)
logger = logging.getLogger(__name__)

# Initialize model service
model_service = ModelService()


@classifier_bp.route('/classify', methods=['POST'])
def classify_bottle():
    """
    Endpoint to classify bottle status (open/closed)

    Accepts:
        - POST request with image file in form data (key: 'image')
    Returns:
        - JSON response with bottle status and confidence score
        - Error message if request is invalid
    """
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({
                "error": "No image file provided"
            }), 400

        image_file = request.files['image']

        # Check if a file was actually selected
        if image_file.filename == '':
            logger.error("No selected image file")
            return jsonify({
                "error": "No selected image file"
            }), 400

        # Validate image file
        if not validate_image(image_file):
            logger.error(f"Invalid image format: {image_file.filename}")
            return jsonify({
                "error": "Invalid image format. Allowed formats: jpg, jpeg, png"
            }), 400

        # Process image and get prediction
        try:
            status, confidence = model_service.predict(image_file)

            # Log successful prediction
            logger.info(f"Successfully classified image: {status} ({confidence:.2f})")

            # Return result in specified format
            return jsonify({
                "Bottle_Status": status,
                "Confidence": float(confidence)
            }), 200

        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            return jsonify({
                "error": "Error processing image"
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in classify endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500