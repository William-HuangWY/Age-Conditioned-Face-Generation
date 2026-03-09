import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FaceAgeDataset(Dataset):
    """
    Dataset for age-conditioned face generation.

    Directory structure:
    root_dir/
        0/
        1/
        ...
        110/
    """

    def __init__(self, root_dir='./datasets/face_age', target_size=128, padding=True, condition_dim=7, transform=None):
        """
        root_dir: dataset root
        target_size: target image size (H=W=target_size)
        padding: True = pad to target size, False = resize
        condition_dim: number of age buckets for one-hot
        transform: optional torchvision transform
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.padding = padding
        self.condition_dim = condition_dim
        self.transform = transform
        self.samples = [] # [(img_path, age), ...]
        self.IMG_EXTS = {".png", ".jpg", ".jpeg"}

        # scan dataset
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path): continue

            try: age = int(folder)
            except ValueError: continue

            if age > 80 or age < 12: continue
            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname.lower())[1]
                if ext in self.IMG_EXTS:
                    self.samples.append((os.path.join(folder_path, fname), age))

        # default transform: ToTensor
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def age_to_bucket(self, age) -> int:
        """
        Convert age to bucket index (0..condition_dim-1)
        Adjust ranges according to condition_dim
        """
        if self.condition_dim == 7:  # 7 buckets
            if 12 <= age <= 18: return 0
            elif 19 <= age <= 25: return 1
            elif 26 <= age <= 31: return 2
            elif 32 <= age <= 39: return 3
            elif 40 <= age <= 53: return 4
            elif 54 <= age <= 67: return 5
            elif 68 <= age <= 80: return 6
            else: raise ValueError(f"Age {age} out of range")
        else:
            # default: scale to condition_dim evenly
            bucket = min(age // (100 // self.condition_dim), self.condition_dim - 1)
            return bucket

    def bucket_to_onehot(self, bucket_idx):
        condition = torch.zeros(self.condition_dim, dtype=torch.float32)
        condition[bucket_idx] = 1.0
        return condition

    def _process_image(self, img: Image.Image):
        """Resize or pad image to target_size"""
        if self.padding:
            width, height = img.size
            pad_left = (self.target_size - width) // 2
            pad_top = (self.target_size - height) // 2
            pad_right = self.target_size - width - pad_left
            pad_bottom = self.target_size - height - pad_top
            img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        else:
            img = img.resize((self.target_size, self.target_size))
        
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._process_image(img)

        bucket_idx = self.age_to_bucket(age)
        condition = self.bucket_to_onehot(bucket_idx)

        return img, condition, age
    
if __name__ == "__main__":
    print("Testing FaceAgeDataset...")
