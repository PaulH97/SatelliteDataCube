import os 
import rasterio
import random
import numpy as np

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

def select_patches(global_mask_patches, ratio_classes, seed):
    random.seed(seed)
    class_indices = [idx for idx, patch in enumerate(global_mask_patches) if np.any(patch==1)]
    no_class_indices = [idx for idx, patch in enumerate(global_mask_patches) if not np.any(patch==1)]
    ratio_classes= [i / 100 for i in ratio_classes]
    num_noclass_patches = int((len(class_indices) / ratio_classes[0]) * ratio_classes[1])
    no_class_indices = random.sample(no_class_indices, num_noclass_patches)
    return class_indices + no_class_indices

def create_and_select_patches(satellite_image, patch_size, selected_indices, indices=False):
    X_patches = satellite_image.process_patches(patch_size, source="img", indices=indices)
    y_patches = satellite_image.process_patches(patch_size, source="msk", indices=indices)
    X_selected_patches = [X_patches[idx] for idx in selected_indices]
    y_selected_patches = [y_patches[idx] for idx in selected_indices]
    satellite_image.unload_bands()
    satellite_image.unload_mask()
    return X_selected_patches, y_selected_patches