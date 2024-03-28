import torch
import json
import argparse
import wandb
from models import create_vit_model
from utils import get_device, get_loaders, test


def load_model(model_path, num_classes):
    model = create_vit_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_run_id(project_name, run_name):
    api = wandb.Api()

    # List all runs in the specified project
    runs = api.runs(f"{api.default_entity}/{project_name}")

    # Filter runs by name
    target_run = None
    for run in runs:
        if run.name == run_name:
            target_run = run
            break

    if target_run is not None:
        return target_run.id
    else:
        raise ValueError(f"No run found for name {run_name} in project {project_name}")


def main(args):
    with open(args.config_filepath) as config_file:
        config_file = json.load(config_file)

    # Load the model
    model = load_model(num_classes=args.num_classes, model_path=args.checkpoint_filename)

    device = get_device()

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
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training (default: 8)')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes (default: 7)')
    parser.add_argument('--config_filepath', type=str, default='./config/config.json', help='Path to configuration file (default: ./config/config.json)')
    parser.add_argument('--checkpoint_filename', type=str, default='./weights/best_model.pth', help='Filename to save the best model (default: ./weights/best_model.pth)')
    parser.add_argument('--cm_filename', type=str, default='confusion_matrix', help='Filename to save confusion matrix (default: confusion_matrix.png)')
    parser.add_argument('--run_name', type=str, default='batch8-lr0_0001-dropout0_1', help='Run name of experiment (default: batch8-lr0_0001-dropout0_1)')
    parser.add_argument('--track_experiment', action='store_true')
    
    args = parser.parse_args()

    main(args)