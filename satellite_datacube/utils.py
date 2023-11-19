import os 
import rasterio
import random
import numpy as np
from matplotlib import pyplot as plt
from rasterio.mask import mask

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
 
