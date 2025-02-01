# model/src/train.py
import os
import yaml
import logging
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Filter out warnings
warnings.filterwarnings('ignore')


class BottleDataset(Dataset):
    """Dataset for bottle images"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_map = {'open': 0, 'closed': 1}
        self._load_dataset()

    def _load_dataset(self):
        for class_name in self.class_map:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob('*.[jp][pn][g]'):
                self.images.append(str(img_path))
                self.labels.append(self.class_map[class_name])

        if not self.images:
            raise RuntimeError(f"No images found in {self.data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class BottleClassifier(nn.Module):
    """Bottle classifier model based on ResNet18"""

    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained ResNet18 instead of ResNet50
        self.resnet = models.resnet18(pretrained=True)

        # Freeze early layers
        for param in list(self.resnet.parameters())[:-4]:
            param.requires_grad = False

        # Simpler classification head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ModelTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        self.config = self._load_config(config_path)
        self.device = self._setup_device()

        # Only initialize GradScaler if using CUDA
        self.use_mixed_precision = False
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            self.use_mixed_precision = True
        elif torch.backends.mps.is_available():
            # MPS doesn't support mixed precision training
            self.scaler = None
            self.use_mixed_precision = False

        self.setup_logging()
        self.setup_model()
        self.setup_data()
        self.best_val_acc = 0
        self.patience_counter = 0

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {str(e)}")

    def _setup_device(self):
        """Setup the training device"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU")
        return device

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def setup_model(self):
        """Initialize model, loss function, and optimizer"""
        self.model = BottleClassifier(self.config['model']['num_classes']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Create checkpoint directory
        checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def setup_data(self):
        """Initialize data loaders"""
        train_transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Smaller size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Smaller size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_dir = Path(__file__).parent.parent / 'data' / 'augmented'
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'

        if not train_dir.exists() or not val_dir.exists():
            raise RuntimeError("Data directories not found. Run augment.py first.")

        train_dataset = BottleDataset(train_dir, transform=train_transform)
        val_dataset = BottleDataset(val_dir, transform=val_transform)

        logging.info(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")

        # Optimize batch size and workers
        num_workers = min(os.cpu_count() or 1, 4)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=128,  # Increased batch size
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=128,  # Increased batch size
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best performance
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            logging.info(f"Saved new best model with validation accuracy: {self.best_val_acc:.2f}%")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_mixed_precision:
                # Use mixed precision training only for CUDA
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training for CPU and MPS
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

            # Clear cache periodically if using MPS
            if self.device.type == "mps" and batch_idx % 50 == 0:
                torch.mps.empty_cache()

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validating'):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(self.val_loader), 100. * correct / total

    def train(self):
        """Full training loop"""
        print("\n" + "=" * 50)
        print(f"Starting training on device: {self.device}")
        print("=" * 50 + "\n")

        early_stop_threshold = 0.01  # Stop if improvement is less than 1%
        total_epochs = 30

        for epoch in range(total_epochs):
            print("\n" + "=" * 50)
            print(f"Epoch [{epoch + 1}/{total_epochs}]")
            print("=" * 50)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Print metrics
            print("\nResults:")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

            # More aggressive early stopping
            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch + 1, is_best=True)
                self.patience_counter = 0

                print(f"\n📈 New best validation accuracy: {val_acc:.2f}%")

                # If improvement is minimal, stop training
                if improvement < early_stop_threshold and epoch > 10:
                    print(f"\n⚠️ Minimal improvement ({improvement:.2f}%), stopping early")
                    break
            else:
                self.patience_counter += 1
                print(f"\n⚠️ No improvement. Patience: {self.patience_counter}/5")

            if self.patience_counter >= 5:  # Reduced patience
                print("\n⚠️ Early stopping triggered - No improvement for 5 epochs")
                break

            # Print a separator
            print("\n" + "-" * 50)


def main():
    try:
        trainer = ModelTrainer()
        trainer.train()
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise


if __name__ == '__main__':
    main()