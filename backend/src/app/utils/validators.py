# backend/src/app/utils/validators.py
from PIL import Image
import io
import logging
from app.config import BaseConfig

logger = logging.getLogger(__name__)


def validate_image(file):
    """Validate uploaded image file"""
    try:
        # Check file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file extension: {file.filename}")
            return False

        # Read file content
        file_content = file.read()

        # Check file size
        if len(file_content) > BaseConfig.MAX_CONTENT_LENGTH:
            logger.warning(f"File too large: {len(file_content)} bytes")
            return False

        # Try opening the image to verify it's valid
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()  # Verify it's actually an image

            # Open again for dimension check (verify() closes the file)
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size

            # Check dimensions
            if width > 4096 or height > 4096:
                logger.warning(f"Image dimensions too large: {width}x{height}")
                return False

        except Exception as e:
            logger.warning(f"Invalid or corrupted image file: {str(e)}")
            return False

        # Reset file pointer for future reads
        file.seek(0)
        return True

    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in BaseConfig.ALLOWED_EXTENSIONS