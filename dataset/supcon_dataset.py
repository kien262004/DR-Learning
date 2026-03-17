import os
from PIL import Image
import random
import torch

from base_dataset import BaseDataset



class DouDataset(BaseDataset):
    
    def __init__(self, image_dir, transform=None):
        super(DouDataset, self).__init__(image_dir, transform)
        
    def _get_path(self, id_image):
        idx, tasks = id_image
        selected = random.sample(tasks, 2)
        paths = [os.path.join(self.image_dir, idx, task + '.jpg') for task in selected]
        return paths, selected
    
    def __getitem__(self, idx):
        id_image = self.image_paths[idx]
        paths, labels = self._get_path(id_image)
        
        image_1 = self._load_image(paths[0])
        image_2 = self._load_image(paths[1])
        
        images_1 = [self.transform(image_1), self.transform(image_1)]
        images_2 = [self.transform(image_2), self.transform(image_2)]

        images = images_1 + images_2
        labels = torch.tensor(labels)
        return images, labels
        
        
        
        