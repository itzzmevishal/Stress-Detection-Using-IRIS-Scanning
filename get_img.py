import os
import random

def get_image(folder_path):
    if image_files := [
        f
        for f in os.listdir(folder_path)
        if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))
    ]:
        return random.choice(image_files)
    else:
        return None
