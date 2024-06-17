from satellite_datacube.image_light import Sentinel1, Sentinel2
from satellite_datacube.utils_light import process_patch, extract_ndvi_values
from satellite_datacube.utils_light import extract_transform, update_patch_spatial_ref, set_nan_value
from pathlib import Path
import xarray as xr
import rasterio 
import dask
import shutil
import random
import geopandas as gpd
from typing import List, Dict, Tuple
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import numpy as np 

class Sentinel2DataCube():
    def __init__(self, base_folder, load=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
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
        for satellite_image_folder in self.si_folder.iterdir():
            if satellite_image_folder.is_dir():
                image_instance = Sentinel2(folder=satellite_image_folder)
                images.append(image_instance)
        images.sort(key=lambda image: image.date)
        return images

    def _load_data(self):
        # This need to be images where the bands are already stacked
        s2_files = [s2.path for s2 in self.images if s2.path.exists()]
        self.data = xr.open_mfdataset(
            s2_files,
            engine='netcdf4',
            chunks="auto",
            parallel=True, # use dask 
            combine="nested",
            concat_dim="time")
    
        # for var in self.data.data_vars:
        #     self.data[var] = self.data[var].astype(np.float16)

        return self.data

    def add_mask(self, mask_file):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": self.data.y.values,"x": self.data.x.values})
        self.data["MASK"] = mask_da
        return self

    def calculate_ndvi(self):
        nir = self.data['B08']
        red = self.data['B04']
        ndvi = (nir - red) / (nir + red)
        self.data['NDVI'] = ndvi
        self.data['NDVI'] = self.data['NDVI'].fillna(0)
        
        return self

    def apply_landslide_dating(self, ann_gdf: gpd.GeoDataFrame) -> List[Dict[str, List[float]]]:

        if "NDVI" not in self.data:
            self.calculate_ndvi()

        if ann_gdf.crs != self.data.rio.crs:
            ann_gdf = ann_gdf.to_crs(self.data.rio.crs)

        from dask.distributed import Client
        client = Client()
        print(client.dashboard_link)

        tasks = []
        for idx, ann in ann_gdf[:2].iterrows():
            print(f"Processing {ann['id']}...")
            delayed_obj = extract_ndvi_values(ann, self.data)
            tasks.append(delayed_obj)
        
        import pdb; pdb.set_trace()

        with ProgressBar():
            results = dask.compute(tasks[0])
        
        return results

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

    def create_patches_fast(self, patch_size:int, patch_indices:dict, chunk_size:int, reset=False) -> List[str]:
        patch_folder = Path(self.base_folder) / "patches" / self.satellite
        if reset and patch_folder.exists():
            shutil.rmtree(patch_folder)
        
        patch_indices = patch_indices["Annotation"] + patch_indices["No-Annotation"]

        S2_xds = self.data
        S2_xds_transform = extract_transform(S2_xds)
        S2_xds_spatial_ref = S2_xds.spatial_ref.copy()
        S2_xds_x, S2_xds_y = S2_xds.x.size, S2_xds.y.size
        
        patch_files = []
        total_patches = len(patch_indices)
        for start in range(0, total_patches, chunk_size):
            print(f"Processing chunk {start} to {min(start + chunk_size, total_patches)}")
            end = min(start + chunk_size, total_patches)
            chunk = patch_indices[start:end]
            
            tasks = []
            for patch_idx in tqdm(chunk, desc="Create patches of chunk"):
                i, j = patch_idx
                filename = Path(patch_folder, f"patch_{i}_{j}.nc")
                if not filename.exists():
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    # min() makes sure that the available index range of the dataset is not exceeded
                    patch = S2_xds.isel(x=slice(i, min(i + patch_size, S2_xds_x)), y=slice(j, min(j + patch_size, S2_xds_y)))
                    if patch.x.size < patch_size or patch.y.size < patch_size:
                        # patch = pad_patch(patch, patch_size) # Fix issues later
                        continue
                    patch = update_patch_spatial_ref(patch, i, j, patch_size, S2_xds_transform, S2_xds_spatial_ref)
                    patch = set_nan_value(patch, fill_value=-9999)         
                    delayed_obj = patch.to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
                    patch_files.append(filename)
                    tasks.append(delayed_obj)
            if tasks:
                print("Saving patches")
                with ProgressBar():
                    dask.compute(*tasks)

        return patch_files

class Sentinel1DataCube():
    def __init__(self, base_folder, load=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
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
        for satellite_image_folder in self.si_folder.iterdir():
            if satellite_image_folder.is_dir():
                image_instance = Sentinel1(folder=satellite_image_folder)
                images.append(image_instance)
        images.sort(key=lambda image: image.date)
        return images

    def _load_data(self):
        # This need to be images where the bands are already stacked
        s1_files = [s1.path for s1 in self.images if s1.path.exists()]
        self.data = xr.open_mfdataset(
            s1_files,
            engine='netcdf4',
            chunks="auto",
            parallel=True, # use dask 
            combine="nested",
            concat_dim="time")
        
        # for var in self.data.data_vars:
        #     self.data[var] = self.data[var].astype(np.float16)
        
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
    
    def create_patches_fast(self, patch_size:int, patch_indices:dict, chunk_size:int, reset=False) -> List[str]:
        patch_folder = Path(self.base_folder) / "patches" / self.satellite
        if reset and patch_folder.exists():
            shutil.rmtree(patch_folder)
        
        patch_indices = patch_indices["Annotation"] + patch_indices["No-Annotation"]

        S1_xds = self.data
        S1_xds_transform = extract_transform(S1_xds)
        S1_xds_spatial_ref = S1_xds.spatial_ref.copy()
        S1_xds_x, S1_xds_y = S1_xds.x.size, S1_xds.y.size

        total_patches = len(patch_indices)
        for start in range(0, total_patches, chunk_size):
            print(f"Processing chunk {start} to {min(start + chunk_size, total_patches)}")
            end = min(start + chunk_size, total_patches)
            chunk = patch_indices[start:end]
            
            tasks = []
            for patch_idx in tqdm(chunk, desc="Create patches of chunk"):
                i, j = patch_idx
                filename = Path(patch_folder, f"patch_{i}_{j}.nc")
                if not filename.exists():
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    # min() makes sure that the available index range of the dataset is not exceeded
                    patch = S1_xds.isel(x=slice(i, min(i + patch_size, S1_xds_x)), y=slice(j, min(j + patch_size, S1_xds_y)))
                    if patch.x.size < patch_size or patch.y.size < patch_size:
                        # patch = pad_patch(patch, patch_size) # Fix issues later
                        continue
                    patch = update_patch_spatial_ref(patch, i, j, patch_size, S1_xds_transform, S1_xds_spatial_ref)
                    patch = set_nan_value(patch, fill_value=-9999)  
                    
                    delayed_obj = patch.to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
                    tasks.append(delayed_obj)
            if tasks:
                print("Saving patches")
                with ProgressBar():
                    dask.compute(*tasks)