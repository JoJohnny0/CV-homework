"""
Module containing functions to load the MVTech dataset.

Useful functions:
- get_loaders
"""

from pathlib import Path
from typing import Literal, TypeAlias

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torchvision import transforms
from torchvision.io import read_image

from modules import globals


ProductType: TypeAlias = Literal['bottle',
                                 'cable',
                                 'capsule',
                                 'carpet',
                                 'grid',
                                 'hazelnut',
                                 'leather',
                                 'metal_nut',
                                 'pill',
                                 'screw',
                                 'tile',
                                 'toothbrush',
                                 'transistor',
                                 'wood',
                                 'zipper'
                                 ]


def load_images(img_dir: Path, mask_dir: Path|None = None) -> tuple[torch.Tensor]|tuple[torch.Tensor, torch.Tensor]:
    """
    Load images from a directory and transform them to match the model input size.
    If no mask directory is provided, only the images are loaded.
    """

    resize_dim: int = 550
    crop_dim: int = 512

    # transofrmations for the images and masks
    img_transform: transforms.Compose = transforms.Compose([transforms.Resize(resize_dim),
                                                            transforms.CenterCrop(crop_dim),
                                                            transforms.ConvertImageDtype(torch.float)
                                                            ])
    mask_transform: transforms.Compose = transforms.Compose([transforms.Resize(resize_dim, interpolation = transforms.InterpolationMode.NEAREST),
                                                             transforms.CenterCrop(crop_dim),
                                                             transforms.ConvertImageDtype(torch.float)
                                                             ])
    
    # loop through the images in the subfolders
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for subfolder in img_dir.iterdir():
        for image in subfolder.iterdir():

            # load the image
            img: torch.Tensor = read_image(str(image))
            img = img_transform(img)
            images.append(img)

            # load the mask
            if mask_dir:
                mask: torch.Tensor
                if subfolder.name == 'good':
                    # if the image is good, create a mask of zeros
                    mask = torch.zeros((1, crop_dim, crop_dim), dtype = img.dtype)
                else:
                    mask_path: Path = mask_dir / subfolder.name / f'{image.stem}_mask{image.suffix}'
                    mask = read_image(str(mask_path))
                    mask = mask_transform(mask)
                masks.append(mask)
    
    # stack into tensors
    if mask_dir:
        return torch.stack(images), torch.stack(masks)
    else:
        return (torch.stack(images),)

def get_loaders(product: ProductType, batch_size: int = 1, val_split: float = 0.2) -> tuple[DataLoader[tuple[torch.Tensor]],
                                                                                            DataLoader[tuple[torch.Tensor]],
                                                                                            DataLoader[tuple[torch.Tensor, torch.Tensor]]
                                                                                            ]:
    """
    Get the train, validation and test data loaders for the MVTech dataset. Only the test loader has masks.
    """

    # directories
    data_dir: Path = globals.MVTECH_DIR
    train_dir: Path = data_dir / product / 'train'
    test_dir: Path = data_dir / product / 'test'
    mask_dir: Path = data_dir / product / 'ground_truth'

    # create the datasets
    train_val_set: TensorDataset = TensorDataset(*load_images(train_dir))
    test_set: TensorDataset = TensorDataset(*load_images(test_dir, mask_dir = mask_dir))

    # split train and validation sets
    train_set: Subset[tuple[torch.Tensor]]
    val_set: Subset[tuple[torch.Tensor]]
    train_set, val_set = random_split(train_val_set, (1 - val_split, val_split))    # type: ignore
    
    # create the data loaders
    train_loader: DataLoader[tuple[torch.Tensor]] = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader: DataLoader[tuple[torch.Tensor]] = DataLoader(val_set, batch_size = batch_size)
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(test_set, batch_size = batch_size)  # type: ignore

    return train_loader, val_loader, test_loader
