import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ISIC2018Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        
        # Convert one-hot encoded labels to single integer labels
        one_hot_label = self.data_frame.iloc[idx, 1:].values
        label = np.argmax(one_hot_label)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
