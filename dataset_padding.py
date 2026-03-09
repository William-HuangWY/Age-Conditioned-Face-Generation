import os
from PIL import Image, ImageOps

input_base = "./datasets/face_age"
output_base = "./datasets/padding_face_aging"
# output_base = "/Users/yingying330/Desktop/face_aging/Age-Conditioned-Face-Generation/datasets/padding_face_aging"

target_size = 256

for folder in os.listdir(input_base):

    input_folder = os.path.join(input_base, folder)
    output_folder = os.path.join(output_base, folder)

    if os.path.isdir(input_folder):
        os.makedirs(output_folder, exist_ok=True)
        for file in os.listdir(input_folder):

            if file.endswith(".png") or file.endswith(".jpg"):

                img_path = os.path.join(input_folder, file)
                img = Image.open(img_path)

                width, height = img.size

                pad_left = (target_size - width) // 2
                pad_top = (target_size - height) // 2
                pad_right = target_size - width - pad_left
                pad_bottom = target_size - height - pad_top

                padded_img = ImageOps.expand(
                    img,
                    (pad_left, pad_top, pad_right, pad_bottom),
                    fill=0
                )

                save_path = os.path.join(output_folder, file)
                padded_img.save(save_path)

print("All images padded and saved to padding_face_aging")
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceAgingDataset(Dataset):
    def __init__(self, root_dir, target_size=256, padding=True, transform=None):
        """
        root_dir: 資料集根目錄，例如 face_age
        target_size: padding/resize 的目標大小
        padding: True 使用 padding, False 使用 resize
        transform: 可額外傳入 torchvision transforms
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.padding = padding
        self.transform = transform
        
        self.image_paths = []
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith(".png") or file.endswith(".jpg"):
                        self.image_paths.append(os.path.join(folder_path, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.padding:
            width, height = img.size
            pad_left = (self.target_size - width) // 2
            pad_top = (self.target_size - height) // 2
            pad_right = self.target_size - width - pad_left
            pad_bottom = self.target_size - height - pad_top
            img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        else:
            img = img.resize((self.target_size, self.target_size))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img