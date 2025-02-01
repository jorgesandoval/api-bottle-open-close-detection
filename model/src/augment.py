# model/src/augment.py
import os
import yaml
import logging
from pathlib import Path
import random
from PIL import Image, ImageEnhance
import albumentations as A
import numpy as np
from tqdm import tqdm


class BottleDataAugmenter:
    def __init__(self, config_path=None):
        if config_path is None:
            # Try to find config file relative to this script
            current_dir = Path(__file__).parent
            config_path = current_dir / 'config.yaml'

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        self.config = self._load_config(config_path)
        if self.config is None:
            raise ValueError("Failed to load configuration")

        self.setup_logging()

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Ensure required configurations exist
            required_configs = ['logging', 'data', 'augmentation']
            for config_section in required_configs:
                if config_section not in config:
                    raise KeyError(f"Required configuration section '{config_section}' not found in config file")

            return config
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return None

    def setup_logging(self):
        try:
            # Create logs directory in the same directory as the script
            script_dir = Path(__file__).parent.parent  # Go up one level to model/
            log_dir = script_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_dir / 'augmentation.log'),
                    logging.StreamHandler()
                ]
            )

            # Test logging
            logging.info("Logging setup completed successfully")

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def extract_top_portion(self, image):
        """Extract the top 30% of the image."""
        width, height = image.size
        crop_height = int(height * self.config['augmentation']['top_crop_percent'])
        return image.crop((0, 0, width, crop_height))

    def apply_augmentations(self, image):
        """Apply all augmentations using albumentations."""
        aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=self.config['augmentation']['horizontal_flip_prob']),
            A.RandomBrightnessContrast(
                brightness_limit=[x - 1 for x in self.config['augmentation']['brightness_range']],
                contrast_limit=[x - 1 for x in self.config['augmentation']['contrast_range']],
                p=0.7
            ),
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(),
                A.GridDistortion(),
            ], p=0.2),
        ])

        # Convert PIL image to numpy array for albumentations
        image_np = np.array(image)
        augmented = aug(image=image_np)
        return Image.fromarray(augmented['image'])

    def augment_single_image(self, image_path, save_dir, num_augmentations=None):
        """Apply augmentation pipeline to a single image."""
        if num_augmentations is None:
            num_augmentations = self.config['augmentation']['num_augmentations']

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.extract_top_portion(image)

            # Create save directory if it doesn't exist
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save preprocessed original
            base_filename = Path(image_path).stem
            original_save_path = save_dir / f"{base_filename}_orig.jpg"
            image.save(original_save_path, quality=95)

            # Generate augmented versions
            for i in range(num_augmentations):
                aug_image = self.apply_augmentations(image)
                aug_save_path = save_dir / f"{base_filename}_aug_{i}.jpg"
                aug_image.save(aug_save_path, quality=95)

            return True

        except Exception as e:
            logging.error(f"Error augmenting image {image_path}: {str(e)}")
            return False

    def augment_dataset(self):
        """Augment entire dataset."""
        try:
            # Get the absolute path to the data directory
            script_dir = Path(__file__).parent.parent  # Go up one level to model/
            data_dir = script_dir / 'data'

            raw_dir = data_dir / 'raw'
            augmented_dir = data_dir / 'augmented'

            if not raw_dir.exists():
                raise FileNotFoundError(f"Raw data directory not found at: {raw_dir}")

            # Create augmented directories
            for split in ['train', 'val', 'test']:
                for class_name in ['open', 'closed']:
                    (augmented_dir / split / class_name).mkdir(parents=True, exist_ok=True)

            # Process each class
            for class_name in ['open', 'closed']:
                class_dir = raw_dir / class_name
                if not class_dir.exists():
                    logging.error(f"Directory not found: {class_dir}")
                    continue

                # Get all images in the class directory
                images = list(class_dir.glob('*.[jp][pn][g]'))
                if not images:
                    logging.error(f"No images found in {class_dir}")
                    continue

                logging.info(f"Found {len(images)} images in {class_dir}")

                # Split images into train/val/test
                random.shuffle(images)
                num_images = len(images)
                num_train = int(0.7 * num_images)
                num_val = int(0.15 * num_images)

                train_images = images[:num_train]
                val_images = images[num_train:num_train + num_val]
                test_images = images[num_train + num_val:]

                # Process each split
                splits = {
                    'train': train_images,
                    'val': val_images,
                    'test': test_images
                }

                for split_name, split_images in splits.items():
                    logging.info(f"Processing {split_name} split for {class_name} class")
                    save_dir = augmented_dir / split_name / class_name

                    # Use more augmentations for training set
                    num_aug = self.config['augmentation']['num_augmentations'] if split_name == 'train' else 2

                    for img_path in tqdm(split_images, desc=f"{split_name}/{class_name}"):
                        success = self.augment_single_image(img_path, save_dir, num_aug)
                        if not success:
                            logging.warning(f"Failed to augment {img_path}")

        except Exception as e:
            logging.error(f"Error during dataset augmentation: {str(e)}")
            raise


def main():
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config.yaml'

        print(f"Looking for config file at: {config_path}")

        augmenter = BottleDataAugmenter(config_path)
        augmenter.augment_dataset()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == '__main__':
    main()