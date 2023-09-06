import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
from .utils import patchify
from .image import Sentinel2
import random

# TODO:

# - change to recalculate the timeseries whenever it is called - remove loading from json file
# -> if someone wants to save the calculation of TS they can provide the idx values of satellite images and give it to the function
# -> that means if they want to save calculation time they can do the json stuff by themself outside the datacube! - show in tutorial

# - change parameters...do not use timeseries length/patch_size because it is not a specific value of the data-cube 
# - use these parameters when the functions are called!!

# - build temp_folder when init?

class SatelliteDataCube:
    def __init__(self, base_folder, load_data=True):

        self.base_folder = base_folder 
        self.satellite_images = self.load_satellite_images()
        self.masks = self.load_masks()
        self.global_mask = self.load_global_mask()
        self.global_mask = self.global_mask if self.global_mask is not None else self.create_global_mask()
        self.seed = 42
    
        self._print_initialization_info()
      
    def _print_initialization_info(self) -> None:
        """
        Display detailed initialization information about the data-cube.

        This utility method prints out a summary of the initialization parameters for the data-cube.
        It shows information about the base folder, desired timeseries length, bad pixel limit 
        for satellite images in the timeseries, and the patch size.

        The printout is structured for clarity, with dividers separating different sections 
        and clear labeling for each parameter.

        Returns:
        None

        Example Output:
        ------------------- base_folder_name -------------------
        Initializing data-cube with following parameter:
        - base folder: .../SatelliteDataCube/Chimanimani
        - Start-End: 2018-09-06 00:00:00 -> 2019-09-16 00:00:00
        - Length of data-cube: 90
        """
        divider = "-" * 20
        #print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print(f"{2*divider}")
        print("Initialized data-cube with following parameter:")
        print(f"- base folder: {self.base_folder}")
        print(f"- Start-End: {next(iter(self.satellite_images.values())).date} -> {next(reversed(self.satellite_images.values())).date}")
        print(f"- Length of data-cube: {len(self.satellite_images)}")
        print(f"{2*divider}")

    def load_global_mask(self):
        global_mask_path = os.path.join(self.base_folder, "global_mask.npy")
        if os.path.exists(global_mask_path):
            return np.load(global_mask_path, allow_pickle=True)
        else:
            print(f"Could not find global_mask file in: {self.base_folder}", "If you want to create a global mask for the timeseries please use create_global_mask().")
            return None 

    def load_patches(self, patch_size, timeseries_length, patches_folder=""):

        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        patches = {}
        for source in ["img", "msk", "msk_gb"]:
            patchPath = os.path.join(patches_folder, f"{source}_patches{patch_size}_ts{timeseries_length}.npy")
            if os.path.exists(patchPath):
                patches[source] = np.load(patchPath)
        if patches:
            print(f"Loaded patches as dictonary with keys [img, msk, msk_gb] from {patches_folder}")
            [print(source, patch.shape) for source, patch in patches.items()]
        else:
            print(f"Could not find patches in: {patches_folder}", "Please use create_patches() to create the necessary patches.")
        return patches

    def load_single_timeseries(self, timeseries_length, ts_folder=""):
        ts_folder = os.path.join(self.base_folder, "selected_timeseries") if not ts_folder else ts_folder
        selected_timeseries = os.path.join(ts_folder, f"ts_{timeseries_length}.json")
        if os.path.exists(selected_timeseries):
            idx_timeseries = [timestep[0] for timestep in selected_timeseries]
            si_timeseries = [self.satellite_images[idx] for idx in idx_timeseries] 
            return si_timeseries
        else:
            print(f"Could not find timeseries in: {ts_folder}", "Please use create_timeseries() to create the necessary timeseries.")
            return []
        
    def load_satellite_images(self):
        """
        Initialize satellite images by scanning the base folder and processing valid satellite image directories.

        This function scans the base folder for directories with numeric names, assuming each such directory 
        corresponds to a satellite image. For every valid directory found, an instance of the `Sentinel2` class 
        is created to represent the satellite image, and the resulting instances are stored in a dictionary 
        where the keys are the indices and the values are the `Sentinel2` instances.

        Note:
        The function assumes that valid satellite image directories in the base folder have numeric names.

        Returns:
        - dict: A dictionary where keys are indices of the satellite images and values are corresponding 
                `Sentinel2` instances representing the images.

        Example:
        Given a base folder structure like:
        base_folder/
        ├── 20180830/
        ├── 20180911/
        ├── temp/
        Calling `load_satellite_images()` will only process the '20180830' and '20180911' directories and return a dictionary 
        with their corresponding `Sentinel2` instances.
        """
        return {
            i: Sentinel2(si_folder)
            for i, si_folder in enumerate(
                si_folder.path 
                for si_folder in os.scandir(self.base_folder) 
                if os.path.isdir(si_folder.path) and si_folder.name.isdigit()
            )
        }

    def load_masks(self):
        masks = {}
        if self.satellite_images:
            for idx, si in self.satellite_images.items():
                masks[idx] = si.initiate_mask()
            return masks
        else:
            print("Satellite images of datacube are not initialized. Please use load_satellite_images() before running this function.")
            return None   

    def create_global_mask(self, save=True, output_folder=""):
        global_masks = []
        if self.masks:
            for mask in self.masks.values():
                if np.any(mask >= 1): 
                    mask_bool = mask >= 1
                    global_masks.append(mask_bool)
            global_mask = np.logical_or.reduce(global_masks).astype(int)
            if save: 
                self.save_global_mask(global_mask=global_mask, output_folder=output_folder)
            return global_mask
        else:
            print("Masks of satellite images are not initialized. Please use load_masks() before calling this function.")
            return None

    def save_global_mask(self, global_mask, output_folder=""):
        if not output_folder:
            output_folder = self.base_folder
        output_file = os.path.join(output_folder, "global_mask.npy")
        with open(output_file, 'wb') as f:
            np.save(file=f,arr=global_mask)
        return

    def create_timeseries(self, timeseries_length, save=True, ts_folder=""):

        def _get_image_by_index(idx):
            return list(self.satellite_images.values())[idx]

        def _is_image_quality_acceptable(image, bad_pixel_limit=15):
            image.calculate_bad_pixels()
            return image._badPixelRatio <= bad_pixel_limit and image not in timeseries

        def _find_acceptable_neighbor(target_idx, search_limit=5):
            """Search for the nearest good quality image before and after the current index. 
            If none is found, return the one with the least bad pixels from the search range."""
            max_index = len(self.satellite_images.values()) - 1
            potential_alternatives = []

            # Search within the range for acceptable images
            for offset in range(1, search_limit + 1):
                for direction in [-1, 1]:
                    new_idx = target_idx + (direction * offset)
                    if 0 <= new_idx <= max_index:
                        neighbor = _get_image_by_index(new_idx)
                        bad_pixel_ratio = neighbor.calculate_bad_pixels()
                        if _is_image_quality_acceptable(neighbor):
                            return neighbor
                        potential_alternatives.append((neighbor, bad_pixel_ratio))

            potential_alternatives.sort(key=lambda x: x[1])  # Sorting by bad pixel ratio
            return potential_alternatives[0][0] if potential_alternatives else None

        print(f"Selecting timeseries with {timeseries_length} satellite images of data-cube")
        timeseries = []
        max_index = len(self.satellite_images.values()) - 1
        selected_indices = np.linspace(0, max_index, timeseries_length, dtype=int)
        for target_idx in selected_indices:
            print("[" + " ".join(str(x) for x in range(len(timeseries) + 1)) + "]", end='\r')
            satellite_image = _get_image_by_index(target_idx)
            if _is_image_quality_acceptable(satellite_image):
                timeseries.append(satellite_image)
            else:
                acceptable_neighbor = _find_acceptable_neighbor(target_idx)
                if acceptable_neighbor:
                    timeseries.append(acceptable_neighbor)
        timeseries = sorted(timeseries, key=lambda image: image.date)
        if save:
            self.save_timeseries(timeseries=timeseries, ts_folder=ts_folder)
        return timeseries

    def save_timeseries(self, timeseries, ts_folder=""):
        tsIdx = [idx for idx, si in self.satellite_images.items() if si in timeseries]
        tsDate = [int(si.date.strftime("%Y%m%d")) for si in timeseries]
        timeseries = [list(item) for item in zip(tsIdx, tsDate)]
        if not ts_folder:
            ts_folder = os.path.join(self.base_folder, "selected_timeseries")
            os.makedirs(ts_folder)
        ts_js_file = os.path.join(ts_folder, f"ts_{len(timeseries)}.json")
        with open(ts_js_file, 'w') as file:
            json.dump(timeseries, file)

    def create_patches(self, source, patch_size, selected_timeseries=[], indices=False):
        print(f"Generating patches from source {source}. Each patch has a size of {patch_size}x{patch_size}px.")
        if not selected_timeseries:
            print(f"No timeseries was selected. Patches of all satellite images of data-cube are created.")
            selected_timeseries = [si for si in self.satellite_images.values()]
        src_patches = [] 
        for image in selected_timeseries:
            print(f"Start with {source} on date {image.date}.")
            patches = image.process_patches(patch_size, source=source, indices=indices) # returns list of patches
            src_patches.append(patches)
            image.unload_bands()
            image.unload_mask()
        # convert it to an array of pattern NxTxCxHxW where N is the number of satellite images in timeseries
        patches[source] = np.swapaxes(np.array(src_patches),0,1) 
        return patches
    
    def filter_patches(self, patches, class_values, class_ratio=(100,0)):
        random.seed(self.seed)
        patch_size = next(iter(patches.values())).shape[-1]
        class_ratio= [i / 100 for i in class_ratio]
        global_mask_patches = patchify(self.global_mask, patch_size)
        class_indices = [idx for idx, patch in enumerate(global_mask_patches) if np.any(np.isin(patch, class_values))]
        no_class_indices = [idx for idx, patch in enumerate(global_mask_patches) if not np.any(np.isin(patch, class_values))]
        num_noclass_patches = int((len(class_indices) / class_ratio[0]) * class_ratio[1])
        no_class_indices = random.sample(no_class_indices, num_noclass_patches)
        selected_patchIDs = class_indices + no_class_indices
        for source, patchArray in patches.items():
            patches[source] = patchArray[selected_patchIDs]  
        return patches
    
    def save_patches(self, patches_folder=""):
        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        for source, patchArray in self.patches.items():
            print(f"Saving patches of source {source} as array with shape {patchArray.shape}")  
            np.save(os.path.join(patches_folder, f"{source}_patches{self.patch_size}_ts{self.timeseries_length}.npy"), patchArray)
        return
    
    def process_patches(self, sources, class_values, seed, indices=False, patches_folder=""):
        self.load_or_built_timeseries()
        patches_idx = self.select_patches(class_values=class_values, seed=seed)
        for source in sources:
            if source in ["img", "msk"]:
                self.create_patches(source=source, indices=indices) # need to get better performance -> takes some RAM...
            else:
                global_mask_patches = patchify(self.global_mask, self.patch_size)         
                self.patches[source] = np.array(global_mask_patches)
        self.filter_patches(patches_idx=patches_idx)
        self.save_patches()
        return self.patches
           	
    def sanity_check(self):
        
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
        idx = random.randint(0, next(iter(self.patches.values())).shape[0]-1)
        # idx = random.randint(0, self.patches["img"].shape[0]-1)
        img = np.moveaxis(self.patches["img"][idx,...],1,-1)
        msk = np.moveaxis(self.patches["msk"][idx,...],1,-1)
        msk_gb = np.moveaxis(self.patches["msk_gb"][idx,...],0,-1)

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

    def create_spectral_signature(self, shapefile, save_csv=True):
        
        from rasterio.mask import mask
        import pandas as pd 

        geometries = []
        # Open your shapefile
        #file = ogr.Open(shapefile)
        layer = file.GetLayer(0)
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            geometry = json.loads(feature.GetGeometryRef().ExportToJson())
            geometries.append(geometry)
        
        file = None
        spectral_sig = {}
        
        for satellite_image in self.satellite_images.values():
           
            satellite_image.initiate_bands()
        
            for b_name, band in satellite_image._bands.items():
                                
                band_value = 0
                
                with rasterio.open(band.path) as src:
                    
                    # Loop over polygons and extract raster values
                    for polygon in geometries:
                        
                        out_image, out_transform = mask(src, [polygon], crop=True)                         
                        band_value += np.mean(out_image)

                mean_value = band_value/len(geometries)
                
                if b_name not in spectral_sig:

                    spectral_sig[b_name] = [mean_value] # {B02:[1,2,3,4,5...90], B03:[1,2,3,4,5...90],...}
                else:
                    spectral_sig[b_name].append(mean_value)

            satellite_image.unload_bands()

        spectral_sig["Timestamp"] = [satellite_image.date for satellite_image in self.satellite_images.values()]
        spectral_sig_df = pd.DataFrame(spectral_sig)
        spectral_sig_df = spectral_sig_df.set_index('Timestamp').reset_index()

        spectral_sig_df.to_csv("spectral_signature.csv")

        return spectral_sig_df  
    
    def plot_band_signature(self, spectral_signature):
        # spectral_signature["NDVI"] = (spectral_signature["B08"] - spectral_signature["B04"])/(spectral_signature["B08"] + spectral_signature["B04"])
        # plt.plot(spectral_signature['Timestamp'], spectral_signature["NDVI"], label="NDVI", marker='o')
        plt.figure(figsize=(10, 8))
        for column in spectral_signature.columns[1:]:  # Assuming first column is 'Timestamp'"B02_DWN_log_[10, 90]_histo.png"
            plt.plot(spectral_signature['Timestamp'], spectral_signature[column], label=column)
        plt.xlabel('Year')
        plt.ylabel('Change')
        plt.title('Annual Change of Different Bands')
        plt.legend()
        plt.show()
        return
