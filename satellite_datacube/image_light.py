from pathlib import Path
from datetime import datetime
import rioxarray
from satellite_datacube.utils_light import extract_S2_band_name, extract_S1_band_name, resample_band, extract_band_number, normalize
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import rasterio
import dask

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

    def stack_bands(self, resolution=10, reset=False):
        try:
            if reset and self.path.exists():
                self.path.unlink()
                print(f"Removed existing file: {self.path}")

            if not self.path.exists():
                # Use Dask to parallelize the processing of each band
                band_names = []
                tasks = []
                for band_name, band_path in self.band_files.items():
                    band_names.append(band_name)
                    tasks.append(dask.delayed(resample_band)(band_path, resolution)) 

                resampled_bands = dask.compute(*tasks)

                # Stack the rasters using Dask
                stacked_raster = xr.concat(resampled_bands, dim='band')
                stacked_raster = stacked_raster.assign_coords(band=band_names)

                # Save the stacked raster using Dask
                stacked_raster.rio.to_raster(self.path, compute=True)
                print(f"Stacked bands successfully and saved under: {self.path}")
            else:
                print(f"File already exists: {self.path}")

            return self
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        stacked_raster = rioxarray.open_rasterio(self.path).chunk({"x": 1000, "y": 1000})
        ds = xr.Dataset()
        for i, band_name in enumerate(self.band_files.keys()):
            ds[band_name] = stacked_raster.isel(band=i).drop_vars('band')
        return ds
    
    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta.copy()
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
        try:
            if reset and self.path.exists():
                self.path.unlink()
                print(f"Removed existing file: {self.path}")

            if not self.path.exists():
                # Use Dask to parallelize the processing of each band
                band_names = []
                tasks = []
                for band_name, band_path in self.band_files.items():
                    band_names.append(band_name)
                    tasks.append(dask.delayed(resample_band)(band_path, resolution)) 

                resampled_bands = dask.compute(*tasks)

                # Stack the rasters using Dask
                stacked_raster = xr.concat(resampled_bands, dim='band')
                stacked_raster = stacked_raster.assign_coords(band=band_names)

                # Save the stacked raster using Dask
                stacked_raster.rio.to_raster(self.path, compute=True)
                print(f"Stacked bands successfully and saved under: {self.path}")
            else:
                print(f"File already exists: {self.path}")

            return self
        except Exception as e:
            print(f"An error occurred: {e}")
            raise


    def load_data(self):
        stacked_raster = rioxarray.open_rasterio(self.path).chunk({"x": 1000, "y": 1000})
        ds = xr.Dataset()
        for i, band_name in enumerate(self.band_files.keys()):
            ds[band_name] = stacked_raster.isel(band=i).drop_vars('band')
        return ds
        
    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta.copy()
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
