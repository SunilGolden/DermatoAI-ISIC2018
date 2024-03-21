import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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
        label = self.data_frame.iloc[idx, 1:].astype('float').to_numpy()

        if self.transform:
            image = self.transform(image)

        return image, label
