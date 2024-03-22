import random
import torch
import numpy as np


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
