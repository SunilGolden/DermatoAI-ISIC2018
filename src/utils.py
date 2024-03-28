import os
import random
import torch
import numpy as np
import wandb
from PIL import Image
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
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


def ensure_folder_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train(model,
          train_loader, 
          val_loader,
          device, 
          num_epochs=100,
          lr=0.0001, 
          weight_decay=0.01, 
          step_size=20, 
          gamma=0.1,
          patience=10, 
          checkpoint_filename='./weights/best_model.pth',
          track_experiment=False):
    ensure_folder_exists(os.path.dirname(checkpoint_filename))
    
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float('inf')
    patience_counter = 0

    print('Starting training...')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, total_samples = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            train_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / total_samples

        # Validation phase
        model.eval()
        val_loss, val_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / total_samples

        print(f'[Epoch {epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # Learning rate scheduling
        scheduler.step()
        if (epoch+1) % step_size == 0:
            print(f'Scheduler step: Learning rate adjusted to {scheduler.get_last_lr()[0]}')

        # Track Experiment
        if track_experiment:
            wandb.log({
                'train/epoch': epoch+1,
                'train/train_acc': train_accuracy,
                'train/train_loss': train_loss,
                'val/val_acc': val_accuracy,
                'val/val_loss': val_loss,
            })

        # Check for early stopping and save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'Saving new best model at epoch {epoch+1}...')
            torch.save(model.state_dict(), checkpoint_filename)
        else:
            patience_counter += 1
            print(f'Validation loss did not improve. Patience counter {patience_counter}/{patience}.')
            if patience_counter >= patience:
                print('Early stopping triggered. Exiting training...')
                break
    print('Training completed.')


def save_confusion_matrix(cm, class_names, cm_filename='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(cm_filename)


def test(model, test_loader, device, class_names, cm_filename='confusion_matrix.png', track_experiment=False):
    model.eval()
    test_loss, test_correct, total_samples = 0, 0, 0
    all_preds = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predictions = torch.max(outputs, 1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = test_correct / total_samples

    # Convert lists to numpy arrays for metric calculations
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute the metrics
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print Metrics
    print('Test Loss: {:.4f}'.format(test_loss))
    print('Test Accuracy: {:.4f}'.format(test_accuracy))
    print('Test F1 Score: {:.4f}'.format(test_f1))
    print('Test Precision: {:.4f}'.format(test_precision))
    print('Test Recall: {:.4f}'.format(test_recall))

    # Save confusion matrix
    save_confusion_matrix(cm, class_names, cm_filename)

    # Track Experiment
    if track_experiment:
        wandb.log({
            'test/test_acc': test_accuracy,
            'test/test_loss': test_loss,
            'test/f1_score': test_f1,
            'test/precision': test_precision,
            'test/recall': test_recall,
        })

        wandb.log({
            'test/confusion_matrix' : wandb.plot.confusion_matrix(probs=None,
                                                             y_true=all_labels, 
                                                             preds=all_preds,
                                                             class_names=class_names)
        })
