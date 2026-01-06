from torch.utils.data import Dataset
import torch
import random
import torchvision.transforms.functional as TF

class MyDataset(Dataset):
    def __init__(self, X, y, transform=None, perc_transform=50):
        self.X = X
        self.y = y
        self.transform = transform
        self.perc_transform = perc_transform  # Percentage of samples to apply transformation


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        #print(f"Shape of x: {x.shape}, Shape of y: {y.shape}", flush=True)

        assert x.shape[-2:] == y.shape[-2:], "Spatial dimensions of image and mask must match"

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Add channel dimension if necessary
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Shape: [1, H, W]
        if len(y.shape) == 2:
            y = y.unsqueeze(0)    # Shape: [1, H, W]

        # Apply transformation based on percentage
        if self.transform and random.uniform(0, 100) < self.perc_transform:
            x, y = self.transform(x, y)

        return x, y

