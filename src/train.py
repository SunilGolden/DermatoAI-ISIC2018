import argparse
import json
import torch
import torch.nn as nn
import wandb
from utils import reset_random, get_loaders, get_device, train
from models import create_vit_model, create_resnet50_model


def main(args):
    with open(args.config_filepath) as config_file:
        config_file = json.load(config_file)

    device = get_device()

    # Data Loaders
    reset_random(config_file['random_seed'])
    train_loader, val_loader, test_loader = get_loaders(config_file, args.batch_size, subset=args.subset)

    # Create model
    reset_random(config_file['random_seed'])

    if args.architecture == 'resnet50':
        model = create_resnet50_model(num_classes=config_file['num_classes'], dropout_rate=args.dropout_rate).to(device)
    elif args.architecture == 'vit':
        model = create_vit_model(num_classes=config_file['num_classes'], dropout_rate=args.dropout_rate).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    if torch.cuda.device_count() > 1:
        print("Available GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    # Track Experiment
    if args.track_experiment:
        WANDB_API_KEY = config_file['wandb']['api_key']
        wandb.login(key=WANDB_API_KEY)

        wandb.init(
            project="DermatoAI-ISIC2018",
            name=args.run_name,
            config = {
                "architecture": args.architecture,
                "random_seed": config_file['random_seed'],
                "batch_size": args.batch_size,
                "num_classes": config_file['num_classes'],
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "dropout_rate": args.dropout_rate,
                "weight_decay": args.weight_decay,
                "step_size": args.step_size,
                "gamma": args.gamma,
                "patience": args.patience,
                "subset": args.subset,
                "config_filepath": args.config_filepath,
                "checkpoint_filename": args.checkpoint_filename
            }
        )

    # Train the model
    reset_random(config_file['random_seed'])
    train(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        patience=args.patience,
        checkpoint_filename=args.checkpoint_filename,
        track_experiment=args.track_experiment)
    
    if args.track_experiment:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model on ISIC2018 Dataset')
    parser.add_argument('--architecture', type=str, default='resnet50', choices=['vit', 'resnet50'], help='Architecture to use (vit or resnet50) (default: resnet50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for LR scheduler (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for LR scheduler (default: 0.1)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (default: 10)')
    parser.add_argument('--subset', type=int, default=None, help='Use a subset of the full dataset (default: None)')
    parser.add_argument('--config_filepath', type=str, default='./config/config.json', help='Path to configuration file (default: ./config/config.json)')
    parser.add_argument('--run_name', type=str, default='resnet-batch8-lr0_0001-dropout0_1', help='Run name of experiment (default: resnet-batch8-lr0_0001-dropout0_1)')
    parser.add_argument('--checkpoint_filename', type=str, default='./weights/best_model.pth', help='Filename to save the best model (default: ./weights/best_model.pth)')
    parser.add_argument('--track_experiment', action='store_true')
    
    args = parser.parse_args()

    main(args)