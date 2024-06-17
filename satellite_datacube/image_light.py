from pathlib import Path
from datetime import datetime
from satellite_datacube.utils_light import extract_S2_band_name, extract_S1_band_name, resample_band, extract_band_number, extract_metadata, normalize, build_xr_dataset
import xarray as xr
import rioxarray as rxr
import numpy as np
from matplotlib import pyplot as plt
import dask

class Sentinel2():
    def __init__(self, folder):
        self.folder = Path(folder)
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.path = self.folder / f"{self.name}.nc"
        
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

    def reset(self):
        if self.path.exists():
            self.path.unlink()
            print(f"Removed existing file: {self.path}")

    def resample_bands(self, resolution):
        band_names = []
        tasks = []
        for band_name, band_path in self.band_files.items():
            band_names.append(band_name)
            tasks.append(dask.delayed(resample_band)(band_path, resolution))

        resampled_bands = dask.compute(*tasks)
        return band_names, resampled_bands

    def create_dataset(self, band_names, resampled_bands):
        image = xr.concat(resampled_bands, dim='band')
        image = image.assign_coords(band=band_names)
        ds = image.to_dataset(name='band_data')
        date_np = np.datetime64(self.date).astype('datetime64[ns]')
        ds = ds.expand_dims(time=[date_np])
        for i, band_name in enumerate(band_names):
            ds[band_name] = ds['band_data'].isel(band=i).astype(np.int16) # still need to fix the data type
        ds = ds.drop_vars('band_data')
        ds = ds.drop_vars('band')
        return ds

    def stack_bands(self, resolution=10, reset=False):
        # maybe implement the option to create a dask delayed object for the whole process and then compute it outside this class
        try:
            if reset:
                self.reset()
            if not self.path.exists():
                band_names, resampled_bands = self.resample_bands(resolution)
                ds = self.create_dataset(band_names, resampled_bands)
                ds.to_netcdf(self.path, compute=True)
                print(f"Stacked bands successfully and saved under: {self.path}")
            else:
                print(f"File already exists: {self.path}")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        return xr.open_dataset(self.path)
    
    def get_metadata(self):
        if self.path.exists():
            image = self.load_data()
            meta = image["spatial_ref"].attrs
            return meta
        else:
            return None

    def calculate_ndvi(self, save=False, reset=False): 
        if self.path.exists():
            image = self.load_data()
            nir = image['B08']
            red = image['B04']
            ndvi = (nir - red) / (nir + red)
            image['NDVI'] = ndvi
            if save:
                ndvi_path = self.path.parent / f"{self.name}_NDVI.tif"
                if reset and ndvi_path.exists():
                    ndvi_path.unlink()
                ndvi.rio.to_raster(ndvi_path, driver='GTiff')
                self.band_files['NDVI'] = ndvi_path
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
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.path = self.folder / f"{self.name}.tif"

    def _initialize_name(self):
        raster_filepath = Path(next(iter(self.band_files.values()))) # s1a_iw_nrb_20170219t095846_015351_0192dd_20ppc-20ppb_vv_dsc.tif
        name_parts = raster_filepath.stem.split("_")
        return "_".join(name_parts[:-2])
    
    def _initialize_date(self):
        date_str = self.name.split("_")[3]
        date = datetime.strptime(date_str, "%Y%m%dt%H%M%S").date()
        return date 
    
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

                ds = stacked_raster.to_dataset(name='band_data')
                date_np = np.datetime64(self.date).astype('datetime64[ns]')
                ds = ds.expand_dims(time=[date_np])
                band_names = self.band_files.keys() # ["VV", "VH"]                    
                for i, band_name in enumerate(band_names):
                    ds[band_name] = ds['band_data'].isel(band=i).drop_vars('band')
                ds = ds.drop_vars('band_data')
                ds.to_netcdf(self.path, compute=True)
                print(f"Stacked bands successfully and saved under: {self.path}")
            else:
                print(f"File already exists: {self.path}")

            return self
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        return xr.open_dataset(self.path)
            
    def get_metadata(self):
        if self.path.exists():
            image = self.load_data()
            meta = image["spatial_ref"].attrs
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
