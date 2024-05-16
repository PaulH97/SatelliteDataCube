from pathlib import Path
from datetime import datetime
import rioxarray
from satellite_datacube.utils_light import extract_S2_band_name, extract_S1_band_name, resample_raster, extract_band_number, normalize
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

class Sentinel2():
    def __init__(self, folder):
        self.folder = Path(folder)
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.path = self.folder / f"{self.name}.tif"
        
    def _initialize_name(self):
        filepath = next(iter(self.band_files.values()))
        file_name = filepath.stem
        return "_".join(file_name.split("_")[:-1])
    
    def _initialize_date(self):
        date_str = self.name.split("_")[-1]
        date = datetime.strptime(date_str, "%Y%m%d").date()
        return date 
    
    def _initialize_bands(self):
        bands = {}
        for tif_file in self.folder.glob('*.tif'):
            band_name = extract_S2_band_name(tif_file)
            if band_name:
                bands[band_name] = tif_file
        sorted_keys = sorted(bands.keys(), key=extract_band_number)
        bands = {k: bands[k] for k in sorted_keys}
        return bands

    def stack_bands(self, resolution=10):
        if not self.path.exists():
            resampled_rasters = []
            band_names = []

            for band_name, band_path in self.band_files.items():
                with rioxarray.open_rasterio(band_path) as raster:
                    if raster.rio.resolution() != (resolution, resolution):
                        raster = resample_raster(raster, resolution)
                    resampled_rasters.append(raster)
                    band_names.append(band_name)
                                  
            stacked_raster = xr.concat(resampled_rasters, dim='band')
            stacked_raster = stacked_raster.assign_coords(band=band_names)
            stacked_raster = stacked_raster.chunk("auto")
            stacked_raster.rio.to_raster(self.path)
        
        return self

    def load_data(self):
        stacked_raster = rioxarray.open_rasterio(self.path).chunk({"x": 1000, "y": 1000})
        ds = xr.Dataset()
        for i, band_name in enumerate(self.band_files.keys()):
            ds[band_name] = stacked_raster.isel(band=i).drop_vars('band')
        return ds
    
    def extract_spatial_metadata(self):
        if self.path.exists():
            raster = rioxarray.open_rasterio(self.path)
            meta = {
                "crs": raster.rio.crs,
                "transform": raster.rio.transform(),
                "shape": raster.rio.shape,
                "bounds": raster.rio.bounds()
                }
            return meta
        else:
            return None

    def calculate_ndvi(self): 
        if self.path.exists():
            image = self.load_data()
            nir = image['B08']
            red = image['B04']
            ndvi = (nir - red) / (nir + red)
            image['NDVI'] = ndvi
            return image

    def plot(self, band_name):
        image = self.load_data()
        if image is None:
            raise ValueError("No data available. Please run 'stack_bands' first.")
        if band_name not in image:
            raise ValueError(f"Band {band_name} is not available in the dataset.")

        image[band_name].plot(figsize=(10, 10), cmap='gray')
        plt.title(f"{band_name} for {self.name}")
        plt.axis('off')
        plt.show()

    def plot_rgb(self):
        if self.path.exists():
            image = self.load_data()
            red = image['B04'].values
            green = image['B03'].values
            blue = image['B02'].values
        
            red_norm = normalize(red)
            green_norm = normalize(green)
            blue_norm = normalize(blue)
            
            # Stack bands to form an RGB image
            rgb = np.dstack((red_norm, green_norm, blue_norm))
            
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.title(f"RGB Image for {self.name}")
            plt.axis('off')
            plt.show()

class Sentinel1():
    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = None
        self.band_files = self._initialize_bands()
        self.date = None   
        self.orbit_state = None
        self.path = None
        self._initialize_attributes()
        
    def _initialize_attributes(self):
        raster_filepath = Path(next(iter(self.band_files.values()))) # s1a_iw_nrb_20170219t095846_015351_0192dd_20ppc-20ppb_vv_dsc.tif
        name_parts = raster_filepath.stem.split("_")
        self.name = "_".join(name_parts[:-2])
        self.orbit_state = name_parts[-1]
        self.date = datetime.strptime(name_parts[3], "%Y%m%dt%H%M%S").date()
        self.path = self.folder / f"{self.name}.tif"

    def _initialize_bands(self):
        bands = {}
        for tif_file in self.folder.glob('*.tif'):
            band_name = extract_S1_band_name(tif_file)
            if band_name:
                band_name = band_name.upper()
                bands[band_name] = tif_file
        return bands

    def stack_bands(self, resolution=10, reset=False):
        if reset and self.path.exists():
            self.path.unlink()
            print(f"Removed existing file: {self.path}")

        if not self.path.exists():
            resampled_rasters = []
            band_names = []

            for band_name, band_path in self.band_files.items():
                with rioxarray.open_rasterio(band_path) as raster:
                    if raster.rio.resolution() != (resolution, resolution):
                        raster = resample_raster(raster, resolution)
                    resampled_rasters.append(raster)
                    band_names.append(band_name)
                                  
            stacked_raster = xr.concat(resampled_rasters, dim='band')
            stacked_raster = stacked_raster.assign_coords(band=band_names)
            stacked_raster = stacked_raster.chunk("auto")
            stacked_raster.rio.to_raster(self.path)
            print(f"Stacked bands successfully and saved under: {self.path}")
        else:
            print(f"File {self.path} already exists. Skipping stacking.")
        return self

    def load_data(self):
        stacked_raster = rioxarray.open_rasterio(self.path).chunk({"x": 1000, "y": 1000})
        ds = xr.Dataset()
        for i, band_name in enumerate(self.band_files.keys()):
            ds[band_name] = stacked_raster.isel(band=i).drop_vars('band')
        return ds
        
    def extract_spatial_metadata(self):
        if self.path.exists():
            raster = rioxarray.open_rasterio(self.path)
            meta = {
                "crs": raster.rio.crs,
                "transform": raster.rio.transform(),
                "shape": raster.rio.shape,
                "bounds": raster.rio.bounds()
                }
            return meta
        else:
            return None

    def plot(self, band_name):
        image = self.load_data()
        if image is None:
            raise ValueError("No data available. Please run 'stack_bands' first.")
        if band_name not in image:
            raise ValueError(f"Band {band_name} is not available in the dataset.")

        image[band_name].plot(figsize=(10, 10), cmap='gray')
        plt.title(f"{band_name} for {self.name}")
        plt.axis('off')
        plt.show()
