"""
Module containing functions to load the datasets.

Useful functions:
- get_loaders

Useful types:
- ProductType: TypeAlias for the product names.

Imports: globals
"""

from pathlib import Path
from typing import Literal, TypeAlias

from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize, ToTensor

from modules import globals


ProductType: TypeAlias = Literal[# MVTech products
                                 'bottle',
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
                                 'zipper',

                                 # BTAD products
                                 '01',
                                 '02',
                                 '03'
                                 ]


def load_images(resize_dim: tuple[int, int],
                crop_dim: tuple[int, int],
                img_dir: Path,
                mask_dir: Path|None = None,
                mask_format: str = '{image}.png'
                ) -> tuple[torch.Tensor]|tuple[torch.Tensor, torch.Tensor]:
    """
    Load images from a directory and transform them to match the model input size.

    Args:
        resize_dim: Dimension to resize the images to.
        crop_dim: Dimension to crop the images to after reshaping.
        img_dir: Path to the directory containing the images.
        mask_dir: Path to the directory containing the masks. If None, only images are loaded.
        mask_format: Format string for the mask filenames, where `{image}` will be replaced by the image filename without extension.
    """

    # transofrmations for the images and masks
    img_transform: Compose = Compose([ToTensor(),
                                      Resize(resize_dim),
                                      CenterCrop(crop_dim)
                                      ])
    mask_transform: Compose = Compose([ToTensor(),
                                       Resize(resize_dim, interpolation = InterpolationMode.NEAREST),
                                       CenterCrop(crop_dim)
                                       ])
    
    # loop through the images in the subfolders
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for subfolder in img_dir.iterdir():
        for image in subfolder.iterdir():

            # load the image
            pil_img: Image.Image = Image.open(image).convert('RGB')
            img: torch.Tensor = img_transform(pil_img)  # type: ignore
            images.append(img)

            # load the mask
            if mask_dir:
                mask: torch.Tensor
                if subfolder.name in ('good', 'ok'):
                    # if the image is good, create a mask of zeros
                    mask = torch.zeros((1, *crop_dim), dtype = img.dtype)
                else:
                    mask_path: Path = mask_dir / subfolder.name / mask_format.format(image = image.stem)
                    pil_mask: Image.Image = Image.open(mask_path)
                    mask = mask_transform(pil_mask) # type: ignore
                masks.append(mask)
    
    # stack into tensors
    if mask_dir:
        return torch.stack(images), torch.stack(masks)
    else:
        return (torch.stack(images),)

def get_loaders(dataset: Literal['MVTech', 'BTAD'],
                product: ProductType,
                resize_dim: tuple[int, int],
                crop_dim: tuple[int, int],
                batch_size: int = 1,
                val_split: float = 0.2
                ) -> tuple[DataLoader[tuple[torch.Tensor]], DataLoader[tuple[torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Get the train, validation and test data loaders for the specified dataset. Only the test loader has masks.

    Args:
        dataset: Name of the dataset to load.
        product: Name of the product to load.
        resize_dim: Dimension to resize the images to.
        crop_dim: Dimension to crop the images to after reshaping.
        batch_size: Batch size for the data loaders.
        val_split: Fraction of the training set to use for validation.
    """

    # directories
    data_dir: Path = globals.DATASET_DIR / dataset / product
    train_dir: Path = data_dir / 'train'
    test_dir: Path = data_dir / 'test'
    mask_dir: Path = data_dir / 'ground_truth'

    # create the datasets
    mask_format: str = '{image}_mask.png' if dataset == 'MVTech' else '{image}.png'
    train_val_set: TensorDataset = TensorDataset(*load_images(resize_dim, crop_dim, train_dir))
    test_set: TensorDataset = TensorDataset(*load_images(resize_dim, crop_dim, test_dir, mask_dir = mask_dir, mask_format = mask_format))

    # split train and validation sets
    train_set: Subset[tuple[torch.Tensor]]
    val_set: Subset[tuple[torch.Tensor]]
    train_set, val_set = random_split(train_val_set, (1 - val_split, val_split))    # type: ignore
    
    # create the data loaders
    train_loader: DataLoader[tuple[torch.Tensor]] = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader: DataLoader[tuple[torch.Tensor]] = DataLoader(val_set, batch_size = batch_size)
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(test_set, batch_size = batch_size)  # type: ignore

    return train_loader, val_loader, test_loader
