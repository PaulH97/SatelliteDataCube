from satellite_datacube.image_light import Sentinel1, Sentinel2
from satellite_datacube.utils_light import process_patch, load_file, create_patch
from satellite_datacube.utils_light import extract_transform
from pathlib import Path
import xarray as xr
import rasterio 
import shutil
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

class Sentinel2DataCube():
    def __init__(self, base_folder, load=True, print=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
        if print:
            self._print_initialization_info()

    def _print_initialization_info(self):
        divider = "-" * 20
        dates = [image.date for image in self.images]
        dates.sort()
        #print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print(f"{2*divider}")
        print("Initialized data-cube with following parameter:")
        print(f"- Base folder: {self.base_folder}")
        print(f"- Satellite mission: {self.satellite}")
        print(f"- Start-End: {min(dates)} -> {max(dates)}")
        print(f"- Length of data-cube: {len(self.images)}")
        print(f"{2*divider}")
        del dates

    def _init_images(self):
        images = []
        images_folder = self.si_folder / "images"
        for folder in images_folder.iterdir():
            if folder.is_dir() and "patches" not in folder.stem:
                image_instance = Sentinel2(folder=folder)
                images.append(image_instance)
        images.sort(key=lambda image: image.date)
        return images

    def _load_data(self):
        s2_files = [s2.path for s2 in self.images if s2.path.exists()]
        datasets = [load_file(file) for file in s2_files]
        self.data = xr.concat(datasets, dim='time')
                
        crs = self.data.attrs["spatial_ref"].crs_wkt
        transform = extract_transform(self.data)

        self.data.rio.write_crs(crs, inplace=True)
        self.data.rio.write_transform(transform, inplace=True)

        self.data.attrs.pop("spatial_ref", None)

        return self.data

    def add_mask(self, mask_file):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": self.data.y.values,"x": self.data.x.values})
        self.data["MASK"] = mask_da
        return self

    def select_patches_based_on_ann(self, patch_size, overlap, ratio=0.3, seed=42) -> Dict[str, List[Tuple[int, int]]]:
        # Check if MASK is a data variable
        if "MASK" not in self.data.data_vars:
            mask_file = self.base_folder / "annotations" / f"{self.base_folder.name}_mask.tif"
            self.add_mask(mask_file)
            
        S2_xds = self.data
        raster_width, raster_height = S2_xds.x.size, S2_xds.y.size
        step_size = patch_size - overlap

        patches_idx_by_ann = {"Annotation": [], "No-Annotation": []}
        
        print("Classifying patches based on annotations...")
        for i in tqdm(range(0, raster_width, step_size), desc="Patches along x-axis"):
            for j in range(0, raster_height, step_size):
                patch_x_end = min(i + patch_size, raster_width)
                patch_y_end = min(j + patch_size, raster_height)
                
                patch = S2_xds.isel(x=slice(i, patch_x_end), y=slice(j, patch_y_end))
                
                if patch.MASK.values.sum() != 0:
                    patches_idx_by_ann["Annotation"].append((i, j))
                else:
                    patches_idx_by_ann["No-Annotation"].append((i, j))
                
        # Apply ratio to the number of patches without annotations
        num_ann = len(patches_idx_by_ann["Annotation"])
        num_no_ann_patches = int(num_ann * ratio)
        
        if num_no_ann_patches > len(patches_idx_by_ann["No-Annotation"]):
            raise ValueError("Not enough non-annotated patches available to meet the desired ratio.")

        random.seed(seed)
        patches_idx_by_ann["No-Annotation"] = random.sample(patches_idx_by_ann["No-Annotation"], num_no_ann_patches)

        return patches_idx_by_ann

    def create_patches(self, patch_size, patch_indices, reset=False):
        patch_folder = Path(self.base_folder) / "patches" / self.satellite
        if reset and patch_folder.exists():
            shutil.rmtree(patch_folder)
        
        if not patch_folder.exists():
            S2_xds = self.data
            patch_indices = patch_indices["Annotation"] + patch_indices["No-Annotation"]

            for patch_idx in patch_indices:
                patch_filename = process_patch(patch_idx, S2_xds, patch_size, patch_folder)
                print("Patch saved to:", patch_filename)
        else:
            print("Patches already exist in the folder:", patch_folder)
    
class Sentinel1DataCube():
    def __init__(self, base_folder, orbit, load=True, print=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.orbit = orbit
        self.si_folder = self.base_folder / self.satellite / self.orbit 
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
        if print:
            self._print_initialization_info()

    def _print_initialization_info(self):
        try:
            divider = "-" * 20
            dates = [image.date for image in self.images]
            dates.sort()
            print(f"{2*divider}")
            print("Initialized data-cube with following parameter:")
            print(f"- Base folder: {self.base_folder}")
            print(f"- Satellite mission: {self.satellite}")
            print(f"- Start-End: {min(dates)} -> {max(dates)}")
            print(f"- Length of data-cube: {len(self.images)}")
            print(f"{2*divider}")
            del dates
        except Exception as e:
            print(f"Error during printing initialization info: {e}")

    def _init_images(self):
        images = []
        images_folder = self.si_folder / "images"
        for folder in images_folder.iterdir():
            if folder.is_dir() and "patches" not in folder.stem:
                image = Sentinel1(folder=folder)
                if image.orbit_state == self.orbit:
                    images.append(image)
        images.sort(key=lambda image: image.date)
        return images
    
    def _load_data(self):
        s1_files = [s1.path for s1 in self.images if s1.path.exists()]
        datasets = [load_file(file) for file in s1_files]
        self.data = xr.concat(datasets, dim='time')
                
        crs = self.data.attrs["spatial_ref"].crs_wkt
        transform = extract_transform(self.data)

        self.data.rio.write_crs(crs, inplace=True)
        self.data.rio.write_transform(transform, inplace=True)

        self.data.attrs.pop("spatial_ref", None)

        return self.data

    def add_mask(self, mask_file):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": self.data.y.values,"x": self.data.x.values})
        self.data["MASK"] = mask_da
        return self

    def select_patches_based_on_ann(self, patch_size, overlap, ratio=0.3, seed=42) -> Dict[str, List[Tuple[int, int]]]:
        # Check if MASK is a data variable
        if "MASK" not in self.data.data_vars:
            mask_file = self.base_folder / "annotations" / f"{self.base_folder.name}_mask.tif"
            self.add_mask(mask_file)
            
        S1_xds = self.data
        raster_width, raster_height = S1_xds.x.size, S1_xds.y.size
        step_size = patch_size - overlap

        patches_idx_by_ann = {"Annotation": [], "No-Annotation": []}
        
        print("Classifying patches based on annotations...")
        for i in tqdm(range(0, raster_width, step_size), desc="Patches along x-axis"):
            for j in range(0, raster_height, step_size):
                patch_x_end = min(i + patch_size, raster_width)
                patch_y_end = min(j + patch_size, raster_height)
                
                patch = S1_xds.isel(x=slice(i, patch_x_end), y=slice(j, patch_y_end))
                
                if patch.MASK.values.sum() != 0:
                    patches_idx_by_ann["Annotation"].append((i, j))
                else:
                    patches_idx_by_ann["No-Annotation"].append((i, j))
                
        # Apply ratio to the number of patches without annotations
        num_ann = len(patches_idx_by_ann["Annotation"])
        num_no_ann_patches = int(num_ann * ratio)
        
        if num_no_ann_patches > len(patches_idx_by_ann["No-Annotation"]):
            raise ValueError("Not enough non-annotated patches available to meet the desired ratio.")

        random.seed(seed)
        patches_idx_by_ann["No-Annotation"] = random.sample(patches_idx_by_ann["No-Annotation"], num_no_ann_patches)

        return patches_idx_by_ann

    def create_patches(self, patch_size, patch_indices, reset=False):
        patch_folder = Path(self.base_folder) / "patches" / self.satellite
        if reset and patch_folder.exists():
            shutil.rmtree(patch_folder)
        
        if not patch_folder.exists():
            S1_xds = self.data
            patch_indices = patch_indices["Annotation"] + patch_indices["No-Annotation"]

            for patch_idx in patch_indices:
                patch_filename = process_patch(patch_idx, S1_xds, patch_size, patch_folder)
                print("Patch saved to:", patch_filename)
        else:
            print("Patches already exist in the folder:", patch_folder)
    