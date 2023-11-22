import os
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from .utils import patchify, select_equally_spaced_items
from .image import Sentinel1, Sentinel2
from .annotation import Sentinel2Annotation, Sentinel1Annotation
import random
import pandas as pd
from datetime import datetime
from glob import glob
from pathlib import Path
# TODO:

# - build temp_folder when init?

# fix the indices parameter -> find a nice solution 

# VERY IMPORTANT: We need to add the GT for all of the following si after the event 
# right now only one timestep contains the mask

# TODO: Update the function with storing the shapefile in a varibale -> creating mask with that and not using the annotation.tif?


class SatelliteDataCube:
    def __init__(self):
        self.base_folder = None
        self.satellite = None
        self.satellite_images_by_date = {}
        self.annotation = {}

    def _print_initialization_info(self):
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
        print(f"- Base folder: {self.base_folder}")
        print(f"- Satellite mission: {self.satellite}")
        print(f"- Start-End: {next(iter(self.satellite_images_by_date.values())).date} -> {next(reversed(self.satellite_images_by_date.values())).date}")
        print(f"- Length of data-cube: {len(self.satellite_images_by_date)}")
        print(f"{2*divider}")

    def select_equal_distributed_satellite_images(self, number_of_images):
        satellite_image_dates = [satellite_image.date for satellite_image in self.satellite_images_by_date.keys()]
        satellite_image_dates_equally_distributed = select_equally_spaced_items(satellite_image_dates, number_of_images)
        selected_satellite_images = [self.satellite_images_by_date.get(satellite_image_date) for satellite_image_date in satellite_image_dates_equally_distributed]
        return selected_satellite_images

    def create_and_patches(self, patch_size):      
        satellite_images_patches = [] # list of np.arrays for each image one array
        for date, satellite_image in self.satellite_images_by_date.items():
            print(f"-> Start with satellite image of date {date}")
            patches = satellite_image.create_patches(patch_size) # returns list of patches
            satellite_images_patches.append(np.array(patches)) # NxCxhxW
        satellite_images_patches = np.stack(satellite_images_patches, axis=1) # convert it to an array of pattern NxTxCxHxW 
        return satellite_images_patches

    def create_and_keep_patches_with_annotation(self, patch_size):      
        satellite_images_patches = [] # list of np.arrays for each image one array
        for date, satellite_image in self.satellite_images_by_date.items():
            print(f"-> Start with satellite image of date {date}")
            patches = satellite_image.create_patches(patch_size) # returns list of patches
            self.annotation.
            satellite_images_patches.append(np.array(patches)) # NxCxhxW
        satellite_images_patches = np.stack(satellite_images_patches, axis=1) # convert it to an array of pattern NxTxCxHxW 
        return satellite_images_patches
    
    def filter_patches(self, satellite_images_patches):      
        satellite_images_patches = [] # list of np.arrays for each image one array

        for date, satellite_image in self.satellite_images_by_date.items():
            print(f"-> Start with satellite image of date {date}")
            
            patches = satellite_image.create_patches(patch_size) # returns list of patches
            satellite_images_patches.append(np.array(patches)) # NxCxhxW
        satellite_images_patches = np.stack(satellite_images_patches, axis=1) # convert it to an array of pattern NxTxCxHxW 
        return satellite_images_patches

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
            selected_timeseries = [si for si in self.satellite_images_by_date.values()]
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
    
class Sentinel2DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.satellite_images_by_date = self._load_satellite_images()
        self.annotation = self._load_annotation()
        self._print_initialization_info()
     
    def _load_annotation(self):
        annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
        s2_satellite_image = next(iter(self.satellite_images_by_date.values()))
        return Sentinel2Annotation(s2_satellite_image, annotation_shapefile)
    
    def _load_satellite_images(self):
        satellite_images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                satellite_images_by_date[date_satellite_image] = Sentinel2(satellite_image_folder, date_satellite_image)
                satellite_images_by_date_sorted = dict(sorted(satellite_images_by_date.items()))
                return satellite_images_by_date_sorted

    def find_higher_quality_satellite_image(self, satellite_image, search_limit=5):
        """Search for the nearest good quality image before and after the current date. 
        If none is found, return the one with the least bad pixels from the search range."""
        satellite_images_dates = sorted(self.satellite_images_by_date.keys())
        start_date_idx = satellite_images_dates.index(satellite_image.date)

        alternative_satellite_images = []
        # Search within the range for acceptable images
        for offset in range(1, search_limit + 1):
            for direction in [-1, 1]:
                new_date_idx = start_date_idx + (direction * offset)
                if 0 <= new_date_idx < len(satellite_images_dates):
                    new_date = satellite_images_dates[new_date_idx]
                    neighbor_satellite_image = self.satellite_images_by_date.get(new_date)
                    if neighbor_satellite_image.is_quality_acceptable():
                        return neighbor_satellite_image
                    else:
                        alternative_satellite_images.append((neighbor_satellite_image, neighbor_satellite_image.calculate_bad_pixels()))
                else:
                    continue
        alternative_satellite_images.sort(key=lambda x: x[1])  # Sorting by bad pixel ratio
        return alternative_satellite_images[0][0] if alternative_satellite_images else satellite_image

    def update_satellite_images(self, number_of_images):
        selected_satellite_images = self.select_equal_distributed_satellite_images(number_of_images)
        updated_satellite_images_by_date = {}
        for satellite_image in selected_satellite_images:
            print("[" + " ".join(str(x) for x in range(len(self.selected_satellite_images.keys()) + 1)) + "]", end='\r')
            if satellite_image.is_quality_acceptable():
                updated_satellite_images_by_date[satellite_image.date] = satellite_image
            else:
                neighbour_satellite_image = self.find_higher_quality_satellite_image(satellite_image)
                updated_satellite_images_by_date[neighbour_satellite_image.date] = neighbour_satellite_image
        self.satellite_images_by_date = updated_satellite_images_by_date
        return self.satellite_images_by_date
            
class Sentinel1DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.satellite_images_by_date = self.load_satellite_images()
        self.annotation = self._load_annotation()

    def _load_annotation(self):
        annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
        s2_satellite_image = next(iter(self.satellite_images_by_date.values()))
        return Sentinel2Annotation(s2_satellite_image, annotation_shapefile)
    
    def load_satellite_images(self):
        satellite_images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                satellite_images_by_date[date_satellite_image] = Sentinel1(satellite_image_folder, date_satellite_image)
                satellite_images_by_date_sorted = dict(sorted(satellite_images_by_date.items()))
                return satellite_images_by_date_sorted
            

# # To search for an image by date
# search_date = date(2021, 1, 1)
# satellite_image = satellite_images_by_date.get(search_date)
