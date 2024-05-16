import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.windows import Window
from .utils import pad_patch, get_metadata_of_window, extract_S2band_info, extract_band_number, extract_S1band_info
from .band import SatelliteBand
from pathlib import Path
import json
from datetime import datetime

class SatelliteImage:
    def __init__(self, folder):
        self.folder = folder # folder of all files correlated to the image like bands, patches etc. 
        self.index_functions = None
        self.bands = {} # dict with "all" bands in self.folder - regular bands + indizes + SCL 
        self.name = None # S2_mspc_l2a_20190509_B02 - function that extract this nam
        self.date = None # date of image
        self.path = None  # Path without indices
        
    def __getitem__(self, band_id):
        try:
            raster_filepath = self.bands[band_id]
            if raster_filepath:
                return SatelliteBand(band_id, raster_filepath)
        except KeyError:
            raise ValueError(f"Band id {band_id} not found.")

    def get_number_of_channels(self):
        if self.path:
            with rasterio.open(self.path, "r") as src:
                return src.count

    def find_band_by_res(self, target_res):
        for band_id in self.bands:
            band = self[band_id]  # Use __getitem__ to get the SatelliteBand instance
            resolution = band.get_resolution() # type: ignore # (10,10)
            if resolution[0] == target_res:
                return band
        return None
    
    def create_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        with rasterio.open(self.path) as src:
            step_size = patch_size - overlay
            for i in range(0, src.width,  step_size):
                for j in range(0, src.height, step_size):
                    args = [j, i, patch_size, patch_size]
                    window = Window(*args)
                    patch = src.read(window=window)        
                    # Check if patch needs padding
                    if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        patch = pad_patch(patch, patch_size)
                    elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        continue  # Skip patches that are smaller than patch_size when padding is False
                    # Update metadata for the patch
                    patch_meta = get_metadata_of_window(src, window)
                    patches_folder = output_dir if output_dir else self.folder / "patches" / "IMG"
                    patches_folder.mkdir(parents=True, exist_ok=True)
                    patch_filepath = patches_folder / f"patch_{i}_{j}.tif"
                    with rasterio.open(patch_filepath, 'w', **patch_meta) as dst:
                            dst.write(patch)
        return patches_folder

class Sentinel2(SatelliteImage):
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = Path(folder)
        self.bands = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = datetime.strptime(self.name.split("_")[-1], "%Y%m%d").date()
        self.index_functions = {"NDVI": self.calculate_ndvi, "NDWI": self.calculate_ndwi}
        self.path = self.folder / f"{self.name}.tif"
        
    def _initialize_name(self):
        raster_filepath = next(iter(self.bands.values()))
        band_name = raster_filepath.stem
        return "_".join(band_name.split("_")[:-1])

    def _initialize_bands(self):
        raster_files_dir = self._filter_raster_files()
        return self._sort_raster_filepaths(raster_files_dir)

    def _filter_raster_files(self):
        raster_files_dir = {}
        for raster_filepath in self.folder.glob('*.tif'):
            band_id = extract_S2band_info(raster_filepath.name)
            if band_id:
                raster_files_dir[band_id] = raster_filepath
        return raster_files_dir

    def _sort_raster_filepaths(self, raster_files_dir):
        sorted_keys = sorted(raster_files_dir.keys(), key=extract_band_number)
        return {k: raster_files_dir[k] for k in sorted_keys}

    def load_meta(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                src_meta = src.meta.copy()
            return src_meta
        else:
            print("No stacked image exists. Please use stack_bands() to create such a file.") 
            return None

    def stack_bands(self, resolution=10):
        """
        Stacks all bands and indices (if requested) of the satellite image into a single multi-band raster file.
        Args:
            resolution (int): The target resolution for resampling all bands. Defaults to 10.
            include_indizes (bool): If True, indices like NDVI and NDWI are calculated, added to the bands,
                                    and included in the stacking process. Defaults to False.
        Returns:
            Path: The path to the stacked image file. This will either be the base path or the index path
                of the SatelliteImage object, depending on whether indices were included.      
        Raises:
            FileNotFoundError: If no band matches the desired resolution.
            Exception: For any other unexpected errors during the stacking process.
        """
        # Dynamic path determination based on whether indices are to be included
        if not self.path.exists():
            band_with_desired_res = self.find_band_by_res(resolution)
            if band_with_desired_res:            
                with rasterio.open(band_with_desired_res.path) as src:
                    meta = src.meta.copy()
                    meta['count'] = len(self.bands)
                    meta['dtype'] = 'uint16'
            else:
                print(f"No band with resolution {resolution} found. Only bands with 10/20m resolution are supported.")  

            with rasterio.open(self.path, 'w', **meta) as dst:
                band_descriptions = []
                for i, (band_id, band_filepath) in enumerate(self.bands.items(), start=1):
                    band = SatelliteBand(band_id, band_filepath)
                    band_arr = band.resample(resolution).load_array()
                    dst.write(np.squeeze(band_arr), i)
                    band_descriptions.append(band_id)
                dst.descriptions = tuple(band_descriptions)

        return self

    def calculate_bad_pixel_ratio(self):
        scl_path = self.bands["SCL"]
        if scl_path:
            with rasterio.open(scl_path) as src:
                slc = src.read()
            bad_pixel_count = np.isin(slc, [0,1,2,3,8,9,10,11])
            bad_pixel_ratio = np.mean(bad_pixel_count) * 100
            return bad_pixel_ratio

    # def is_quality_acceptable(self, bad_pixel_ratio=15):
    #     if not self.bad_pixel_ratio:
    #         self.calculate_bad_pixel_ratio()
    #     return self.bad_pixel_ratio < bad_pixel_ratio

    def calculate_all_indizes(self):
        raster_files_dir = self._filter_raster_files()
        if not "NDVI" in raster_files_dir:
            self.calculate_ndvi()
        if not "NDWI" in raster_files_dir:
            self.calculate_ndwi()
        return self

    def calculate_ndvi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        red = SatelliteBand(band_name="B04", band_path=self.bands["B04"])
        nir = SatelliteBand(band_name="B08", band_path=self.bands["B08"])
        red_arr, nir_arr = red.load_array().astype(float), nir.load_array().astype(float)
        index = (nir_arr - red_arr) / (nir_arr + red_arr)
        index = np.nan_to_num(index)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': red.load_meta()["crs"],
                'transform': red.load_meta()["transform"],
                'compress': 'lzw',
                'nodata': np.NaN
            }
            index_path = self.folder / f"{self.name}_NDVI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.bands["NDVI"] = index_path
        return index
    
    def calculate_ndwi(self, save=True):
        nir = SatelliteBand(band_name="B08", band_path=self.bands["B08"])
        swir1 = SatelliteBand(band_name="B11", band_path=self.bands["B11"]).resample(10)
        nir_arr = np.nan_to_num(nir.load_array().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        swir1_arr = np.nan_to_num(swir1.load_array().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        small_value = 1e-10  # Avoid division by zero
        index = (nir_arr - swir1_arr) / (nir_arr + swir1_arr + small_value)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': nir.load_meta()["crs"],
                'transform': nir.load_meta()["transform"],
                'compress': 'lzw',
                'nodata': np.NaN
            }
            index_path = self.folder / f"{self.name}_NDWI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.bands["NDWI"] = index_path
        return index
    
    def plot_rgb(self):
        red = self._loaded_bands["B04"].stretch_contrast()
        green = self._loaded_bands["B03"].stretch_contrast()
        blue = self._loaded_bands["B02"].stretch_contrast()
        rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
        plt.imshow(rgb)
        plt.title(f"RGB of {self.location} on date: {self.date}")
        plt.show()
        plt.savefig("rgb.png")
        return        

class Sentinel1(SatelliteImage):
    def __init__(self, folder):
        super().__init__(folder)
        self.folder = Path(folder)
        self.bands = self._initialize_bands()
        self.name = None
        self.orbit_state = None 
        self.date =  None
        self.path = None
        self.index_functions = {"CR": self.calculate_CR}
        self._initialize_attributes()
        
    def _initialize_attributes(self):
        raster_filepath = Path(next(iter(self.bands.values()))) # s1a_iw_nrb_20170219t095846_015351_0192dd_20ppc-20ppb_vv_dsc.tif
        name_parts = raster_filepath.stem.split("_")
        self.name = "_".join(name_parts[:-2])
        self.orbit_state = name_parts[-1]
        self.date = datetime.strptime(name_parts[3], "%Y%m%dt%H%M%S").date()
        self.path = self.folder / f"{self.name}.tif"
        return 

    def _initialize_bands(self):
        bands_with_path = {}
        for raster_filepath in self.folder.glob('*.tif'):
            band_id = extract_S1band_info(raster_filepath.name)
            if band_id:
                bands_with_path[band_id] = raster_filepath

        sorted_keys = sorted(bands_with_path.keys())
        return {k: bands_with_path[k] for k in sorted_keys}

    def stack_bands(self):
        if not self.path.exists():
            band_path = next(iter(self.bands.values()))
            with rasterio.open(band_path) as src:
                meta = src.meta.copy()
            meta['count'] = len(self.bands)

            with rasterio.open(self.path, 'w', **meta) as dst:
                band_descriptions = []
                for i, (band_id, band_filepath) in enumerate(self.bands.items(), start=1):
                    band = SatelliteBand(band_id, band_filepath)
                    band_array = band.load_array()
                    if band_array is not None:
                        dst.write(np.squeeze(band_array), i)
                        band_descriptions.append(band_id)
                dst.descriptions = tuple(band_descriptions)

    def calculate_CR(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        vv = SatelliteBand(band_name="VV)", band_path=self.bands["VV"])
        vh = SatelliteBand(band_name="VH)", band_path=self.bands["VH"])
        cross_ratio = vh.load_array().astype(float) / vv.load_array().astype(float)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': cross_ratio.shape[1],
                'width': cross_ratio.shape[2],
                'count': 1,
                'dtype': cross_ratio.dtype,
                'crs': vv.meta["crs"],
                'transform': vv.meta["transform"],
                'compress': 'lzw',
                'nodata': None
            }
            index_path = self.folder / f"{self.name}_CR.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(cross_ratio)
            self.bands["CR"] = index_path
        return cross_ratio

    def plot_CR(self):
        red = self._loaded_bands["B04"].stretch_contrast()
        green = self._loaded_bands["B03"].stretch_contrast()
        blue = self._loaded_bands["B02"].stretch_contrast()
        rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
        plt.imshow(rgb)
        plt.title(f"RGB of {self.location} on date: {self.date}")
        plt.show()
        plt.savefig("rgb.png")
        return        
