import os

import numpy as np
from PIL import Image

def save_output_image(image, base_path="outputs", base_filename="inputimage", seed=0):
    """Save an image with a unique filename in the specified directory."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    # Check for existing files and create a new filename
    index = 0
    while True:
        if index == 0:
            filename = f"{base_filename}_seed_{seed}.png"
        else:
            filename = f"{base_filename}_{str(index).zfill(4)}_seed_{seed}.png"
        
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            image.save(file_path)
            break
        index += 1
    return file_path

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask