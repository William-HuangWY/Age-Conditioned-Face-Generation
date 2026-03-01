import kagglehub
import shutil
import os

path = kagglehub.dataset_download("frabbisw/facial-age")
target_dir = os.getcwd() + "/dataset"
os.makedirs(target_dir, exist_ok=True)
shutil.copytree(path, target_dir, dirs_exist_ok=True)
shutil.rmtree(path)