# backend/src/app/main.py
from flask import Flask
from flask_cors import CORS
import logging
from pathlib import Path
import os
import sys

# Add src directory to Python path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from app.routes.classifier import classifier_bp
from app.utils.logger import setup_logger
from app.config import config


def create_app(config_name='default'):
    """Create and configure the Flask application"""
    # Initialize Flask app
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Setup CORS
    CORS(app)

    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info('Starting Bottle Classifier API...')

    # Register blueprints
    app.register_blueprint(classifier_bp, url_prefix='/api')

    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'bottle-classifier-api'}, 200

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        logger.error(f'404 Error: {error}')
        return {'error': 'Resource not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'500 Error: {error}')
        return {'error': 'Internal server error'}, 500

    logger.info('Application startup complete')
    return app


def main():
    """Main entry point for the application"""
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )


if __name__ == '__main__':
    main()