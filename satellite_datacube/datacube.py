import os
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import json
from .utils import patchify
from .image import Sentinel1, Sentinel2, Sentinel12
import random
import pandas as pd
from datetime import datetime
from glob import glob

# TODO:

# - build temp_folder when init?

# fix the indices parameter -> find a nice solution 

# VERY IMPORTANT: We need to add the GT for all of the following si after the event 
# right now only one timestep contains the mask

# TODO: Update the function with storing the shapefile in a varibale -> creating mask with that and not using the annotation.tif?


class SatelliteDataCube:
    def __init__(self, base_folder, satellite):
        self.satellite = satellite    
        self.base_folder = base_folder 
        self.satellite_images = self._load_satellite_images()
        self.labels_shp = glob(os.path.join(self.base_folder, "annotations", "*.shp"))[0]
        self.labels = self._load_labels_as_df()
        self.masks = self._load_masks()
        self.global_mask = self._load_or_create_global_mask()
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
        print(f"- Images from satellite mission: {self.satellite}")
        print(f"- Base folder: {self.base_folder}")
        print(f"- Start-End: {next(iter(self.satellite_images.values())).date} -> {next(reversed(self.satellite_images.values())).date}")
        print(f"- Length of data-cube: {len(self.satellite_images)}")
        print(f"{2*divider}")

    def _load_or_create_global_mask(self):
        global_mask_path = os.path.join(self.base_folder, "other", "global_mask.npy")
        if os.path.exists(global_mask_path):
            global_mask = np.load(global_mask_path, allow_pickle=True)
        else:
            print(f"Could not find global_mask file in: {self.base_folder}. Creating the global mask for the data-cube.")
            global_mask = self._create_global_mask()
        return global_mask

    def _create_global_mask(self, save=True, output_folder=""):
        global_masks = []
        if self.masks:
            for mask in self.masks.values():
                mask_bool = mask >= 1
                if mask_bool.any():  # Checks if any value is True
                    global_masks.append(mask_bool)
            global_mask = np.logical_or.reduce(global_masks).astype(int)
            if save: 
                self._save_global_mask(global_mask=global_mask, output_folder=output_folder)
            return global_mask
        else:
            print("Masks of satellite images are not initialized. Please use load_masks() before calling this function.")
            return None

    def _save_global_mask(self, global_mask, output_folder=""):
        if not output_folder:
            output_folder = os.path.join(self.base_folder, "other")
        output_file = os.path.join(output_folder, "global_mask.npy")
        with open(output_file, 'wb') as f:
            np.save(file=f,arr=global_mask)
        return

    def _load_satellite_images(self):
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
        satellite_folder = os.path.join(self.base_folder, self.satellite)
        satellite_images = {}
        if self.satellite == "sentinel-1":
            for i, si_folder in enumerate(si_folder.path for si_folder in os.scandir(satellite_folder) if os.path.isdir(si_folder.path) and si_folder.name.isdigit()):
                    si_date = datetime.strptime(os.path.basename(si_folder), "%Y%m%d").date()
                    si_location = os.path.basename(self.base_folder)
                    satellite_images[i] = Sentinel1(si_folder=si_folder.path, location=si_location, date=si_date)
            return satellite_images
        elif self.satellite == "sentinel-2":
            for i, si_folder in enumerate(si_folder.path for si_folder in os.scandir(satellite_folder) if os.path.isdir(si_folder.path) and si_folder.name.isdigit()):
                    si_date = datetime.strptime(os.path.basename(si_folder), "%Y%m%d").date()
                    si_location = os.path.basename(self.base_folder)
                    satellite_images[i] = Sentinel2(si_folder=si_folder, location=si_location, date=si_date)
            return satellite_images
        else:
            for i, si_folder in enumerate(si_folder.path for si_folder in os.scandir(satellite_folder) if os.path.isdir(si_folder.path) and si_folder.name.isdigit()):
                    si_date = datetime.strptime(os.path.basename(si_folder), "%Y%m%d").date()
                    si_location = os.path.basename(self.base_folder)
                    satellite_images[i] = Sentinel12(si_folder=si_folder.path, location=si_location, date=si_date)
            return satellite_images
    
    def _load_masks(self):
        masks = {}
        if self.satellite_images:
            for idx, si in self.satellite_images.items():
                masks[idx] = si.initiate_mask()
                si.unload_mask()
            return masks
        else:
            print("Satellite images of datacube are not initialized. Please use load_satellite_images() before running this function.")
            return None   

    def _load_labels_as_df(self):
        labels = gpd.read_file(self.labels_shp)
        labels = pd.DataFrame(labels)
        return labels

    def get_bitemporal_images(self, labels_idx):
        pre_date = self.labels_df.iloc[labels_idx]["pre_date"]
        post_date = self.labels_df.iloc[labels_idx]["post_s2cf"]
        pre_date = datetime.strptime(pre_date, "%Y-%m-%d").date()
        post_date = datetime.strptime(post_date, "%Y-%m-%d").date()
        satellite_images_dates = {idx: satellite_image.date for idx,satellite_image in self.satellite_images.items()}
        pre_closest_key = min(satellite_images_dates.keys(), key=lambda key: abs(pre_date - satellite_images_dates[key]))
        post_closest_key = min(satellite_images_dates.keys(), key=lambda key: abs(post_date - satellite_images_dates[key]))
        pre_image = self.satellite_images[pre_closest_key]
        post_image = self.satellite_images[post_closest_key]
        return pre_image, post_image

    def load_patches(self, patch_size, timeseries_length, patches_folder=""):
        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        patches = {}
        for source in ["images", "masks", "global_mask"]:
            patchPath = os.path.join(patches_folder, f"{source}_patches{patch_size}_ts{timeseries_length}.npy")
            if os.path.exists(patchPath):
                patches[source] = np.load(patchPath)
        if patches:
            print(f"Loaded patches as dictonary with keys [images, masks, global_mask] from {patches_folder}")
            [print(source, patch.shape) for source, patch in patches.items()]
        else:
            print(f"Could not find patches in: {patches_folder}", "Please use create_patches() to create the necessary patches.")
        return patches

    def load_single_timeseries(self, timeseries_length, ts_folder=""):
        ts_folder = os.path.join(self.base_folder, "selected_timeseries") if not ts_folder else ts_folder
        ts_path = os.path.join(ts_folder, f"ts_{timeseries_length}.json")
        if os.path.exists(ts_path):
            selected_timeseries = json.load(open(ts_path))
            idx_timeseries = [timestep[0] for timestep in selected_timeseries]
            si_timeseries = [self.satellite_images[idx] for idx in idx_timeseries] 
            print(f"Found ts {selected_timeseries} in {ts_path}")
            return si_timeseries
        else:
            print(f"Could not find timeseries in: {ts_folder}", "Please use create_timeseries() to create the necessary timeseries.")
            return []

    def create_timeseries(self, timeseries_length):

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
        print("Selected timeseries")
        timeseries = sorted(timeseries, key=lambda image: image.date)
        return timeseries

    def save_timeseries(self, timeseries, ts_folder=""):
        tsIdx = [idx for idx, si in self.satellite_images.items() if si in timeseries]
        tsDate = [int(si.date.strftime("%Y%m%d")) for si in timeseries]
        timeseries = [list(item) for item in zip(tsIdx, tsDate)]
        ts_folder = os.path.join(self.base_folder, "selected_timeseries") if not ts_folder else ts_folder
        if not os.path.exists(ts_folder):
            os.makedirs(ts_folder)
        ts_js_file = os.path.join(ts_folder, f"ts_{len(timeseries)}.json")
        with open(ts_js_file, 'w') as file:
            json.dump(timeseries, file)

    def create_patches(self, patch_size, selected_timeseries=[], indices=False):      
        if not selected_timeseries:
            print(f"No timeseries was selected. Patches of all satellite images of data-cube are created.")
            selected_timeseries = [si for si in self.satellite_images.values()]
        print(f"Generating patches of size {patch_size}x{patch_size}px for {len(selected_timeseries)} satellite images with corresponding masks and for one global mask:") 
        all_src_patches = {}
        for source in ["images", "masks", "global_mask"]:
            if source in ["images", "masks"]:
                src_patches = [] 
                for image in selected_timeseries:
                    print(f"-> Start with satellite {source} of date {image.date}")
                    patches = image.process_patches(patch_size=patch_size, source=source, indices=indices) # returns list of patches
                    src_patches.append(patches)
                    image.unload_bands()
                    image.unload_mask()  
                src_patches = np.swapaxes(np.array(src_patches),0,1) # convert it to an array of pattern NxTxCxHxW 
            else:
                print(f"-> Start with {source}" )
                src_patches = np.array(patchify(source_array=self.global_mask, patch_size=patch_size)) # NxCxHxW
            all_src_patches[source] = src_patches 
        return all_src_patches

    def filter_patches(self, patches, class_values, class_ratio=(100,0)):
        random.seed(self.seed)
        class_ratio= [i / 100 for i in class_ratio]
        patch_size = next(iter(patches.values())).shape[-1]
        global_mask_patches = patchify(self.global_mask,patch_size)
        class_indices = [idx for idx, patch in enumerate(global_mask_patches) if np.any(np.isin(patch, class_values))]
        no_class_indices = [idx for idx, patch in enumerate(global_mask_patches) if not np.any(np.isin(patch, class_values))]
        num_noclass_patches = int((len(class_indices) / class_ratio[0]) * class_ratio[1])
        no_class_indices = random.sample(no_class_indices, num_noclass_patches)
        filtered_patches = {}
        for source, patchArray in patches.items():
            filtered_patches[source] = patchArray[class_indices + no_class_indices] # for masks it can happen that no of the selected masks of si contains GT 
        return filtered_patches

    def save_patches(self, patches, patches_folder=""):
        patches_folder = os.path.join(self.base_folder,"patches") if not patches_folder else patches_folder
        patch_array = next(iter(patches.values()))
        patch_size, ts_length = patch_array.shape[-1], patch_array.shape[1]
        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        print(f"Saving patches in folder: {patches_folder}")
        for source, patchArray in patches.items(): 
            np.save(os.path.join(patches_folder, f"{source}_patches{patch_size}_ts{ts_length}.npy"), patchArray)
            print(f"Saved patches from source {source} as array with shape: {patchArray.shape}") 
        return 

    def process_patches(self, patch_size, class_values, selected_timeseries=[], indices=False, patches_folder=""):
                
        patches = self.create_patches(patch_size=patch_size, selected_timeseries=selected_timeseries, indices=indices)
        [print(f"Created patches of {source.upper()} with shape: {array.shape}") for source, array in patches.items()]
        filtered_patches = self.filter_patches(patches=patches, class_values=class_values)
        [print(f"Filtered patches of {source.upper()} so that the shape changed to: {array.shape}") for source, array in patches.items()]
        self.save_patches(patches=filtered_patches, patches_folder=patches_folder)
        return filtered_patches

    def create_spectral_signature(self, selected_timeseries=[], indices=False, output_folder=""):
        if not selected_timeseries:
            selected_timeseries = [si for si in self.satellite_images.values()]
        spectral_sig = {}
        for image in selected_timeseries:
            image_spectral_sig = image.calculate_spectral_signature(shapefile=self.labels_shp,indices=indices)
            spectral_sig[str(image.date)] = image_spectral_sig
        if output_folder:
            spectral_sig_df = pd.DataFrame(spectral_sig)
            spectral_sig_df.to_csv(os.path.join(output_folder,f"{self.satellite}_spectralSig_ts{len(selected_timeseries)}.csv"))
        return spectral_sig
    
    def plot_spectral_signature(self, spectral_signature, output_folder=""):
        bands = list(spectral_signature[list(spectral_signature.keys())[0]].keys())
        fig, ax = plt.subplots(figsize=(10,6))
        for band in bands:
            time_steps = list(spectral_signature.keys())
            band_values = [spectral_signature[time_step][band] for time_step in time_steps]
            ax.plot(time_steps, band_values, label=band)

        ax.set_title("Spectral Signature over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Band Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        if output_folder:
            plt.savefig(os.path.join(output_folder, f"{self.satellite}_spectralSig_ts{len(time_steps)}.png"))
        return

