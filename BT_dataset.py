from torchvision import transforms
import torch.utils.data as data
import numpy as np
import torch
import skimage.io as io


class Dataset(data.Dataset):
    def __init__(self, image_paths, labels,resize):
        self.paths = image_paths
        self.labels = labels
        self.resize = resize
       
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300,5024)),
            transforms.ToTensor()])
        self.default_transform = transforms.Compose([transforms.ToTensor()]) 
        
    def __len__(self):
        return self.paths.shape[0]
    
    def __getitem__(self, i):
        image_ = io.imread(self.paths[i])
        if(len(image_.shape) < 3):
                image_ = np.stack((image_,)*3, axis=-1)
                
        if(self.resize):
            image = self.resize_transform(image_)
        else:
            image = self.default_transform(image_)
        
        label = torch.zeros(1, image.size(1), image.size(2))
        return image, label
