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