import torch
import os
from PIL import Image

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cvae_model import ConditionalVariationAutoEncoder
from dataset_padding import FaceAgingDataset
class FaceAgeDataset(Dataset):
    """
    
    datasets/face_age/
        0/
        1/
        2/
        ...
        110/
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir, image_size=128, condition_dim=5, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.condition_dim = condition_dim
        self.samples = []

        # 如果外面沒有傳 transform，就使用預設 resize 128 + ToTensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # 這裡就是 resize 128
                transforms.ToTensor(),                        # 轉成 [0,1]
            ])
        else:
            self.transform = transform

        # scan dataset
        for age_folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, age_folder)

            if not os.path.isdir(folder_path):
                continue

            try:
                age = int(age_folder)
            except ValueError:
                continue

            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname.lower())[1]
                if ext not in self.IMG_EXTS:
                    continue

                img_path = os.path.join(folder_path, fname)
                self.samples.append((img_path, age))

    def __len__(self):
        return len(self.samples)

    def age_to_bucket(self, age):
        """
        split age into 8 buckets
        
        """
        if 12<= age <=18:
            return 0
        elif 19 <= age <= 25:
            return 1
        elif 26 <= age <= 31:
            return 2
        elif 32 <= age <= 39:
            return 3
        elif 40 <= age <= 53:
            return 4
        elif 54 <= age <= 67:
            return 5
        elif 68 <= age <= 80:
            return 6
        else:
            return 7

    def bucket_to_onehot(self, bucket_idx):
        condition = torch.zeros(self.condition_dim, dtype=torch.float32)
        condition[bucket_idx] = 1.0
        return condition

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]

        # 讀圖片
        image = Image.open(img_path).convert("RGB")

        # resize 128 + tensor
        image = self.transform(image)

        # age -> bucket -> one-hot
        bucket_idx = self.age_to_bucket(age)
        condition = self.bucket_to_onehot(bucket_idx)

        return image, condition, age
    
    
    