import json
import argparse
import wandb
from utils import get_device, get_loaders, load_model, get_run_id, test


def main(args):
    with open(args.config_filepath) as config_file:
        config_file = json.load(config_file)

    device = get_device()

    # Load the model
    model = load_model(args.architecture, num_classes=args.num_classes, model_path=args.checkpoint_filename, device=device)
    
    _, _, test_loader = get_loaders(config_file, batch_size=args.batch_size)

    # Track Experiment
    if args.track_experiment:
        WANDB_API_KEY = config_file['wandb']['api_key']
        wandb.login(key=WANDB_API_KEY)

        run_id = get_run_id("DermatoAI-ISIC2018", args.run_name)

        wandb.init(
            project="DermatoAI-ISIC2018",
            id=run_id,
            name=args.run_name,
            resume="must"
        )

    test(model, test_loader, device, ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], args.cm_filename, args.track_experiment)

    if args.track_experiment:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--architecture', type=str, default='resnet50', choices=['vit', 'resnet50'], help='Architecture to use (vit or resnet50) (default: resnet50)')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training (default: 8)')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes (default: 7)')
    parser.add_argument('--config_filepath', type=str, default='./config/config.json', help='Path to configuration file (default: ./config/config.json)')
    parser.add_argument('--checkpoint_filename', type=str, default='./weights/best_model.pth', help='Filename to save the best model (default: ./weights/best_model.pth)')
    parser.add_argument('--cm_filename', type=str, default='confusion_matrix', help='Filename to save confusion matrix (default: confusion_matrix.png)')
    parser.add_argument('--run_name', type=str, default='resnet-batch8-lr0_0001-dropout0_1', help='Run name of experiment (default: resnet-batch8-lr0_0001-dropout0_1)')
    parser.add_argument('--track_experiment', action='store_true')
    
    args = parser.parse_args()

    main(args)