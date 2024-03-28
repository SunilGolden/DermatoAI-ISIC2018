# DermatoAI-ISIC2018

# Dataset

ISIC 2018 Task 3 Dataset

**Citation**
HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

**Dataset source:** [ISIC Challenge](https://challenge.isic-archive.com/data/#2018)

<br />

# Usage Guide

1. **Data:** Download the data from https://challenge.isic-archive.com/data/#2018.

1. **Install Python:** Make sure Python is installed on your system. If not, you can download and install Python from the official Python website: https://www.python.org/downloads/

2. **Create a virtual environment:** 

	```bash
	python -m venv env
	```

3. **Activate the virtual environment**

	> For Windows
	```bash
	env\Scripts\activate
	```

	> For macOS/Linux
	```bash
	source env/bin/activate
	```

4. **Install the dependencies**
	
	```bash
	pip install -r requirements.txt
	```

<br />

## Configuration

- Prepare a JSON configuration file with the following structure (an example is provided as `config/config-example.json`):

    ```json
    {
        "random_seed": 42,
        "data_paths": {
            "train": "./data/ISIC2018_Task3_Training_Input",
            "validation": "./data/ISIC2018_Task3_Validation_Input",
            "test": "./data/ISIC2018_Task3_Test_Input"
        },
        "label_paths": {
            "train": "./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
            "validation": "./data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
            "test": "./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
        },
        "wandb": {
            "api_key": "YOUR_WANDB_API_KEY_HERE"
        }
    }
    ```

## Training

```bash
python src\train.py 
        --batch_size <batch_size> 
        --num_classes <num_classes> 
        --epochs <num_epochs> 
        --lr <learning_rate> 
        --dropout_rate <dropout_rate> 
        --weight_decay <weight_decay> 
        --step_size <step_size> 
        --gamma <gamma> 
        --patience <patience> 
        --subset <subset_size> 
        --config_filepath <path_to_config_file> 
        --run_name <experiment_name> 
        --checkpoint_filename <path_to_save_best_model> 
        [--track_experiment]
```

#### Arguments

- `--batch_size` (default: 8): Input batch size for training.
- `--num_classes` (default: 7): Number of classes.
- `--epochs` (default: 100): Number of epochs to train.
- `--lr` (default: 0.0001): Learning rate.
- `--dropout_rate` (default: 0.1): Dropout rate.
- `--weight_decay` (default: 0.01): Weight decay for regularization.
- `--step_size` (default: 20): Step size for the learning rate scheduler.
- `--gamma` (default: 0.1): Multiplicative factor for learning rate decay.
- `--patience` (default: 10): Patience for early stopping.
- `--subset` (default: None): Use a subset of the full dataset for training, specify the subset size.
- `--config_filepath` (default: './config/config.json'): Path to the configuration file.
- `--run_name` (default: 'batch8-lr0_0001-dropout0_1'): Name of the experiment run.
- `--checkpoint_filename` (default: './weights/best_model.pth'): File path to save the best model weights.
- `--track_experiment` (optional): Flag to track the experiment using Weights & Biases.


<br />

## Evaluation

```bash
python src\evaluate.py 
        --batch_size <batch_size> 
        --num_classes <num_classes> 
        --config_filepath <path_to_config_file> 
        --checkpoint_filename <path_to_saved_model> 
        --cm_filename <filename_for_confusion_matrix> 
        --run_name <experiment_name> 
        [--track_experiment]
```

#### Arguments

- `--batch_size` (default: 8): Input batch size for evaluation.
- `--num_classes` (default: 7): Number of classes.
- `--config_filepath` (default: './config/config.json'): Path to the configuration file.
- `--checkpoint_filename` (default: './weights/best_model.pth'): Path to the saved model weights file.
- `--cm_filename` (default: 'confusion_matrix'): Filename (without extension) for saving the generated confusion matrix image.
- `--run_name` (default: 'batch8-lr0_0001-dropout0_1'): Name of the experiment run to be used for identifying the Weights & Biases run ID.
- `--track_experiment` (optional): Flag to track the evaluation using Weights & Biases.

