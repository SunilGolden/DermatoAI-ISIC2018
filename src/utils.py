import random
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from dataset import ISIC2018Dataset


def reset_random(random_seed=42):
    # Set random seed for Python's built-in random module
    random.seed(random_seed)

    # Set random seed for NumPy
    np.random.seed(random_seed)

    # Set random seed for PyTorch CPU
    torch.manual_seed(random_seed)

    # Set random seed for PyTorch CUDA backend (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def get_data_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, resample=False, fillcolor=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def get_loaders(config, batch_size, subset=None):
    # Apply transformations
    data_transforms = get_data_transforms()

    # Create datasets
    train_dataset = ISIC2018Dataset(csv_file=config['label_paths']['train'],
                                    img_dir=config['data_paths']['train'],
                                    transform=data_transforms['train'])

    validation_dataset = ISIC2018Dataset(csv_file=config['label_paths']['validation'],
                                         img_dir=config['data_paths']['validation'],
                                         transform=data_transforms['validation'])

    test_dataset = ISIC2018Dataset(csv_file=config['label_paths']['test'],
                                   img_dir=config['data_paths']['test'],
                                   transform=data_transforms['test'])

    # If a subset value is given, use only that many samples from each dataset
    if subset is not None:
        train_indices = range(min(subset, len(train_dataset)))
        val_indices = range(min(subset, len(validation_dataset)))
        test_indices = range(min(subset, len(test_dataset)))

        train_dataset = Subset(train_dataset, train_indices)
        validation_dataset = Subset(validation_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
