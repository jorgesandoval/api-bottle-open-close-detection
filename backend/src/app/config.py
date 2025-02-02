# src/app/config.py
import os
from pathlib import Path


class BaseConfig:
    """Base configuration"""
    # API Settings
    API_TITLE = 'Bottle Classifier API'
    API_VERSION = 'v1'

    # Server Settings
    HOST = '0.0.0.0'
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = False

    # Model Settings
    MODEL_PATH = Path(__file__).parent / 'models' / 'best_model.pth'
    MODEL_INPUT_SIZE = (160, 160)  # The size your model expects

    # Image Settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Logging Settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DIR = Path(__file__).parent.parent.parent / 'logs'

    # Performance Settings
    BATCH_SIZE = 1
    NUM_WORKERS = 1

    # Image Settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MAX_IMAGE_PIXELS = 4096 * 4096         # Maximum total pixels
    MAX_IMAGE_DIMENSION = 4096             # Maximum width or height

    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        # Create necessary directories
        os.makedirs(BaseConfig.LOG_DIR, exist_ok=True)


class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(BaseConfig):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'

    # Override settings for production
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        # Add production-specific initialization here
        # Example: configure production logging
        import logging
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            filename=BaseConfig.LOG_DIR / 'app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(BaseConfig.LOG_FORMAT))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


class TestingConfig(BaseConfig):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

    # Use test-specific paths
    MODEL_PATH = Path(__file__).parent / 'models' / 'test_model.pth'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}