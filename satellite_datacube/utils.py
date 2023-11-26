import os 
import rasterio
import random
import numpy as np
from matplotlib import pyplot as plt
from rasterio.mask import mask
from rasterio.transform import Affine
from .annotation import SatelliteImageAnnotation

def create_patches_with_metadata(source_array, patch_size, source_array_metadata, padding=True):
    """Utility function to create patches from a source array."""
    num_bands, size_x, size_y = source_array
    patches_with_metadata = []
    for i in range(0, size_x, patch_size):
        for j in range(0, size_y, patch_size):
            patch = source_array[:num_bands, i:i+patch_size, j:j+patch_size]
            # Check if patch needs padding
            if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                patch = pad_patch(patch, patch_size)
            elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                continue  # Skip patches that are smaller than patch_size when padding is False
            # Update metadata for patch
            patch_transform= update_patch_transform(source_array_metadata, i, j)
            patch_metadata = source_array_metadata
            patch_metadata["transform"] = patch_transform
            patch_metadata["count"] = num_bands
            patch_metadata['width'], patch_metadata['height'] = patch_size, patch_size
            patches_with_metadata.append((patch, patch_metadata))
    return patches_with_metadata

def pad_patch(patch, patch_size):
    # Determine how much padding is needed
    pad_x = patch_size - patch.shape[1]
    pad_y = patch_size - patch.shape[2]
    # Here, we're padding with zeros - you can adjust the 'constant_values' parameter as needed for your application
    padded_patch = np.pad(patch, ((0, 0), (0, pad_x), (0, pad_y)), mode='constant', constant_values=0)
    return padded_patch

def update_patch_transform(original_metadata, start_x, start_y):
    original_transform = original_metadata['transform']
    new_transform = Affine.translation(
        original_transform.c + start_x * original_transform.a, 
        original_transform.f + start_y * original_transform.e
        )
    return new_transform

def plot_spectral_signature(spectral_signature):
    band_ids = list(spectral_signature.keys())
    reflectances = list(spectral_signature.values())
    plt.figure(figsize=(10, 5))
    plt.plot(band_ids, reflectances, marker='o', linestyle='-')
    plt.title("Spectral Signature")
    plt.xlabel("Band ID")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.show()
    plt.savefig("Spectral_sig.png")

def band_management_decorator(func):
    """Decorator to manage band loading and unloading."""
    def wrapper(*args, **kwargs):
        self = args[0] # Assuming 'self' is the first argument (as in a method of a class)
        self.load_all_bands()
        try:
            result = func(*args, **kwargs) # Execute the original function
            return result
        finally:
            self.unload_all_bands()
    return wrapper

def is_name_a_number(path):
    try:
        # Attempt to convert the name to a float
        float(path.name)
        return True
    except ValueError:
        return False
    
def contrast_stretching_minmax(image):
    img = image.astype(np.float32)
    img_cs = np.empty_like(img, dtype=np.float32)
    # Perform contrast stretching on each channel
    for band in range(image.shape[-1]):
        img_min = image[...,band].min().astype(np.float32)
        img_max = image[...,band].max().astype(np.float32)
        img_cs[...,band] = (image[...,band] - img_min) / (img_max - img_min)
    return img_cs

def contrast_stretching_precentile(image, percentiles=(2,98)):
    img = image.astype(np.float32)
    img_cs = np.empty_like(img, dtype=np.float32)
    # Perform contrast stretching on each channel
    for band in range(image.shape[-1]):
        img_min = np.percentile(image[...,band], percentiles[0]).astype(np.float32)
        img_max = np.percentile(image[...,band], percentiles[1]).astype(np.float32)
        img_cs[...,band] = (image[...,band] - img_min) / (img_max - img_min)
    return img_cs


# def create_patches(self, patch_size, padding=True):
#     num_bands, size_x, size_y = self.array.shape
#     patches = []
#     for i in range(0, size_x, patch_size):
#         for j in range(0, size_y, patch_size):
#             patch = self.array[:, i:i+patch_size, j:j+patch_size]
#             # Check if patch needs padding
#             if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
#                 patch = pad_patch(patch, patch_size)
#             elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
#                 continue  # Skip patches that are smaller than patch_size when padding is False
#             patches.append(patch)
#     return patches

# def get_patches_metadata(self, patches):
#     num_bands, size_x, size_y = self.array.shape
#     patch_size = patches[0].shape[-1]
#     patches_metadata = []
#     for i in range(0, size_x, patch_size):
#         for j in range(0, size_y, patch_size):
#             # Save the patch using raster
#             patch_transform = update_patch_transform(self.meta, i, j)
#             patch_metadata = self.meta.copy()
#             patch_metadata["transform"] = patch_transform
#             patch_metadata['height'], patch_metadata['width'] = patch_size, patch_size
#             patches_metadata.append(patch_metadata)
#     return patches_metadata

# def save_patches(self, patches, patches_metadata):
#     patches_folder = self.path.parent / "patches"
#     patches_folder.mkdir(parents=True, exist_ok=True)
#     for idx, patch, patch_meta in enumerate(zip(patches, patches_metadata)):
#         patch_meta['driver'] = 'GTiff'
#         patch_path = patches_folder / (self.name + f"patch{idx}.tif" )
#         with rasterio.open(patch_path, 'w', **patch_meta) as dst:
#             dst.write(patch[0, :, :], 1)
#     return patches_folder

def sanity_check(patches):

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

def filter_patches(self, patches):      
    satellite_images_patches = [] # list of np.arrays for each image one array
    for patch in patches:
        print(f"-> Start with satellite image of date {date}")
        patches = satellite_image.create_patches(patch_size) # returns list of patches
        satellite_images_patches.append(np.array(patches)) # NxCxhxW
    satellite_images_patches = np.stack(satellite_images_patches, axis=1) # convert it to an array of pattern NxTxCxHxW 
    return satellite_images_patches

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
    idx = len(os.listdir(patches_folder))

    meta = template_meta.copy()
    meta.update(width=patch_size, height=patch_size, count=patch.shape[0])
    meta['dtype'] = patch.dtype
    transform = template_meta["transform"] * rasterio.Affine.translation(j, i)
    meta.update(transform=transform)

    output_path = os.path.join(patches_folder, f"{idx:05}.tif")
    if os.path.exists(output_path):
        os.remove(output_path)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(patch)
    return patches_folder

def generate_bbox(labels_df, buffer_amount):
    from shapely.geometry import shape, box
    
    def buffered_bbox(polygon, buffer):
        poly_shape = shape(polygon)
        minx, miny, maxx, maxy = poly_shape.bounds
        bbox = box(minx - buffer, 
                   miny - buffer, 
                   maxx + buffer, 
                   maxy + buffer)
        return bbox
    
    labels_df["bbox"] = labels_df["geometry"].apply(lambda polygon: buffered_bbox(polygon, buffer_amount))
    return labels_df

def mask_image(labels_df, satellite_image, output_folder=""):
    satellite_image.initiate_bands()
    for idx, row in labels_df.iterrows():
        bbox = row['bbox']
        out_images = []
        for satellite_band in satellite_image._bands.values():
            if satellite_band.name != "SCL":
                satellite_band = satellite_band.resample(resolution=10, reference_band_path=satellite_image._bands["B02"].path, save_file=True)
                src_band = satellite_band.band
                out_image, out_transform = mask(src_band, [bbox], crop=True)
                out_images.append(out_image)
        
                src_meta = src_band.meta.copy()
                src_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],  
                    "width": out_image.shape[2],   
                    "transform": out_transform, 
                    "dtype": out_image.dtype,      
                    "count": out_image.shape[0]
                })

                if output_folder:
                    label_folder = os.path.join(output_folder, str(idx))
                    os.makedirs(label_folder, exist_ok=True)
                    with rasterio.open(os.path.join(label_folder, f"S2_{satellite_band.name}.tif"), 'w', **src_meta) as dest:
                        dest.write(out_image)      
    satellite_image.unload_bands()
    return

# from rasterio.plot import show
# raster = rasterio.open(os.path.join(label_folder, f"S2_B02.tif"))
# x, y = geometry.exterior.xy
# fig, ax = plt.subplots()
# ax.plot(x, y)
# show(raster, ax=ax, transform=raster.transform)
# plt.show()
 
    # def process_filtered_patches(self, patch_size, indices=False, output_folder=None):
    #     self.stack_bands(indices=indices)
    #     self.initiate_mask()     
    #     stackedBands = np.delete(self._stackedBands, -1, axis=0)
    #     img_patches = patchify(stackedBands, patch_size)
    #     msk_patches = patchify(self._mask, patch_size)
    #     band = next(iter(self.bands.values()))
    #     with rasterio.open(band.path) as src:
    #         template_meta = src.meta
    #     for i, patches in enumerate(zip(img_patches,msk_patches)):
    #         x, y = (i // (self._stackedBands.shape[1] // patch_size)) * patch_size, (i % (self._stackedBands.shape[1] // patch_size)) * patch_size
    #         if np.any(np.isin(patches[1], 1)):
    #             img_patch_folder = save_patch(output_folder, x, y, patches[0], template_meta, patch_size, source="images")
    #             msk_patch_folder = save_patch(output_folder, x, y, patches[1], template_meta, patch_size, source="masks")
    #     self.unload_bands()
    #     self.unload_mask()
    #     return img_patch_folder, msk_patch_folder