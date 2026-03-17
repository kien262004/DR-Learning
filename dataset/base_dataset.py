import os
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from PIL import Image
import random


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        image_dir,
        transform=None,
    ):
        """
        Args:
            image_dir (str): thư mục chứa ảnh
            transform (callable): transform cho ảnh
            return_path (bool): có trả về path hay không
        """

        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self._load_images()

    def _load_images(self):
        """Load toàn bộ path ảnh"""
        image_files = {}
        
        for idx in os.listdir(self.image_dir):
            image_files[idx] = [os.path.splitext(file)[0] for file in  os.listdir(os.path.join(self.image_dir, idx))]            

        image_files = list(image_files.items())
        return image_files

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """Load ảnh từ disk"""
        img = Image.open(path).convert("RGB")
        return img

    @abstractmethod
    def __getitem__(self, idx):
        pass