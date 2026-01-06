from torch.utils.data import Dataset
import torch
import random
import torchvision.transforms.functional as TF

class MyDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.apply_transform = transform  # Rename to avoid confusion with the method name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        assert image.shape == mask.shape, "Image and mask dimensions must match"

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Add channel dimension if necessary
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Shape: [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)    # Shape: [1, H, W]

        # Apply transformations if enabled
        if self.apply_transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
