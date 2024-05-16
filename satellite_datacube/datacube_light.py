from satellite_datacube.image_light import Sentinel1, Sentinel2
from satellite_datacube.utils_light import S1_xds_preprocess, S2_xds_preprocess
from satellite_datacube.utils_light import update_patch_spatial_ref, fill_nans, pad_patch
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import rasterio 
import numpy as np
import shutil

class Sentinel2DataCube():
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
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

        s2_files = [s2.path for s2 in self.images if s2.path.exists()]

        self.data = xr.open_mfdataset(
            s2_files,
            engine="rasterio",
            chunks="auto",
            preprocess=S2_xds_preprocess,
            combine="nested",
            concat_dim="time")
        
        return self.data

    def add_mask(self, mask_file):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": self.data.y.values,"x": self.data.x.values})
        self.data["MASK"] = mask_da
        self.data = self.data.chunk("auto")
        return self

    def calculate_ndvi(self):
        nir = self.data['B08']
        red = self.data['B04']
        ndvi = (nir - red) / (nir + red)
        self.data['NDVI'] = ndvi
        return ndvi

    def create_patches(self):
        pass

    def clean(self):
        pass   

class Sentinel1DataCube():
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
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
            engine="rasterio",
            chunks="auto",
            preprocess=S1_xds_preprocess,
            combine="nested",
            concat_dim="time")
        
        return self.data

    def add_mask(self, mask_file):
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": self.data.y.values,"x": self.data.x.values})
        self.data["MASK"] = mask_da
        self.data = self.data.chunk("auto")
        return self

    def select_patches_based_on_ann(self, patch_size, overlap, ratio=0.3):
        # Check if MASK is a data variable
        if "MASK" not in self.data.data_vars:
            mask_file = self.base_folder / "annotations" / f"{self.base_folder.name}_mask.tif"
            self.add_mask(mask_file)
            
        S1_xds = self.data.chunk({'x': patch_size, 'y': patch_size})
        raster_width, raster_height = S1_xds.x.size, S1_xds.y.size
        step_size = patch_size - overlap

        patches_idx_by_ann = {"Annotation": [], "No-Annotation": []}
        
        print("Classifying patches based on annotations...")
        for i in tqdm(range(0, raster_width, step_size), desc="Patches along x-axis"):
            for j in range(0, raster_height, step_size):
                patch_x_end = min(i + patch_size, raster_width)
                patch_y_end = min(j + patch_size, raster_height)
                
                patch = S1_xds.isel(x=slice(i, patch_x_end),
                                    y=slice(j, patch_y_end))
                
                if patch.MASK.values.sum() != 0:
                    patches_idx_by_ann["Annotation"].append((i, j))
                else:
                    patches_idx_by_ann["No-Annotation"].append((i, j))

        # Apply ratio to the number of patches without annotations
        num_ann = len(patches_idx_by_ann["Annotation"])
        num_no_ann_patches = int(num_ann * ratio)
        
        if num_no_ann_patches > len(patches_idx_by_ann["No-Annotation"]):
            raise ValueError("Not enough non-annotated patches available to meet the desired ratio.")

        selected_patches = np.random.choice(patches_idx_by_ann["No-Annotation"], size=num_no_ann_patches, replace=False).tolist()
        patch_indices = patches_idx_by_ann["Annotation"] + selected_patches
        
        return patch_indices

    def create_patches(self, patch_size, patch_indices, reset=False):
        
        patch_folder = Path(self.base_folder) / "patches" / self.satellite
        if reset and patch_folder.exists():
            shutil.rmtree(patch_folder)

        S1_xds = self.data.chunk({'x': patch_size, 'y': patch_size})
        
        for idx in tqdm(patch_indices, desc="Creating selected patches"):

            i, j = idx
            patch = S1_xds.isel(x=slice(i, min(i + patch_size, S1_xds.x.size)),
                                y=slice(j, min(j + patch_size, S1_xds.y.size)))
            
            patch = pad_patch(patch, patch_size)
            patch = update_patch_spatial_ref(patch, i, j, patch_size, S1_xds)
            patch = fill_nans(patch, fill_value=-9999)
            
            filename = patch_folder / f"patch_{i}_{j}.nc"
            filename.parent.mkdir(parents=True, exist_ok=True)
            patch.to_netcdf(filename)

        # combined = xr.concat(samples, dim="sample")
        # # Reorder dimensions to: (sample, time, band, y, x)
        # combined = combined.transpose('sample', 'time', 'variable', 'y', 'x')

    def clean(self):
        pass   