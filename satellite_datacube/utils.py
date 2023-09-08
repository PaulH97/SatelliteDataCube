import os 
import rasterio
import random
import numpy as np
from matplotlib import pyplot as plt

def patchify(source_array, patch_size):
    """Utility function to create patches from a source array."""
    num_bands, size_x, size_y = source_array.shape
    patches = []
    for i in range(0, size_x, patch_size):
        for j in range(0, size_y, patch_size):
            patch = source_array[:, i:i+patch_size, j:j+patch_size]
            if patch.shape == (num_bands, patch_size, patch_size):
                patches.append(patch)
    return patches

def sanity_check(patches):

    def contrastStreching(image):
        
        image = image.astype(np.float32)
        imgCS = np.empty_like(image, dtype=np.float32)

        # Perform contrast stretching on each channel
        for band in range(image.shape[-1]):
            imgMin = image[...,band].min().astype(np.float32)
            imgMax = image[...,band].max().astype(np.float32)
            imgCS[...,band] = (image[...,band] - imgMin) / (imgMax - imgMin)
        
        return imgCS

    # Pick random sample
    idx = random.randint(0, next(iter(patches.values())).shape[0]-1)
    img = np.moveaxis(patches["images"][idx,...],1,-1)
    msk = np.moveaxis(patches["masks"][idx,...],1,-1)
    msk_gb = np.moveaxis(patches["global_mask"][idx,...],0,-1)

    timesteps = img.shape[0]
    nrows, nclos = 2, timesteps+1
    fig, axs = plt.subplots(nrows=nrows, ncols=nclos, figsize=(28, 2), sharey=True)     
    for i in range(nrows):
        if i == 0: 
            for j in range(timesteps):
                img_data = contrastStreching(img[j,:,:,:3])
                axs[i][j].imshow(img_data)  
                axs[i][j].axis('off')
            
            axs[i][timesteps].imshow(msk_gb, cmap='gray')
            axs[i][timesteps].axis('off')  

        else:
            for j in range(timesteps):
                axs[i][j].imshow(msk[j,...], cmap='gray')  
                axs[i][j].axis('off')
            
            axs[i][timesteps].imshow(msk_gb, cmap='gray')
            axs[i][timesteps].axis('off')  

    plt.show()
    return

def filter_patches(global_mask, patches, class_values, class_ratio=(100,0), seed=42):
    random.seed(seed)
    class_ratio= [i / 100 for i in class_ratio]
    patch_size = next(iter(patches.values())).shape[-1]
    global_mask_patches = patchify(global_mask,patch_size)
    class_indices = [idx for idx, patch in enumerate(global_mask_patches) if np.any(np.isin(patch, class_values))]
    no_class_indices = [idx for idx, patch in enumerate(global_mask_patches) if not np.any(np.isin(patch, class_values))]
    num_noclass_patches = int((len(class_indices) / class_ratio[0]) * class_ratio[1])
    no_class_indices = random.sample(no_class_indices, num_noclass_patches)
    filtered_patches = {}
    for source, patchArray in patches.items():
        filtered_patches[source] = patchArray[class_indices + no_class_indices] # for masks it can happen that no of the selected masks of si contains GT 
    return filtered_patches

def save_patches(patches, patches_folder=""):
    patches_folder = os.path.join(os.getcwd(),"patches") if not patches_folder else patches_folder
    patch_array = next(iter(patches.values()))
    patch_size, ts_length = patch_array.shape[-1], patch_array.shape[1]
    print(patch_size, ts_length)
    if not os.path.exists(patches_folder):
        os.makedirs(patches_folder)
    for source, patchArray in patches.items(): 
        np.save(os.path.join(patches_folder, f"{source}_patches{patch_size}_ts{ts_length}.npy"), patchArray)
        print(f"Saved patches from source {source} as array with shape: {patchArray.shape}") 
    return

def save_patch(output_folder, i, j, patch, template_meta, patch_size, source):
    """Utility function to save a patch to a file."""
    patches_folder = os.path.join(output_folder, "patches", source)
    os.makedirs(patches_folder, exist_ok=True)

    meta = template_meta.copy()
    meta.update(width=patch_size, height=patch_size, count=patch.shape[0])
    meta['dtype'] = patch.dtype
    transform = template_meta["transform"] * rasterio.Affine.translation(j, i)
    meta.update(transform=transform)

    output_path = os.path.join(patches_folder, f'{i}_{j}.tif')
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(patch)
    return