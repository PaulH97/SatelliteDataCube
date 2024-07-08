from pathlib import Path
from datetime import datetime
from satellite_datacube.utils_light import extract_S2_band_name, extract_S1_band_name, resample_raster, extract_band_number, extract_metadata, normalize, build_xr_dataset
import xarray as xr
import rioxarray as rxr
import numpy as np
from matplotlib import pyplot as plt
import dask
import rasterio

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

    def reset(self):
        if self.path.exists():
            self.path.unlink()

    def resample_bands(self, resolution):
        band_names = []
        resampled_bands = []
        for band_name, band_path in self.band_files.items():
            band_names.append(band_name)
            with rxr.open_rasterio(band_path) as raster:
                resampled_raster = resample_raster(raster, resolution)
            resampled_bands.append(resampled_raster)
        return band_names, resampled_bands

    def create_dataset(self, band_names, resampled_bands):
        image = xr.concat(resampled_bands, dim='band')
        image = image.assign_coords(band=band_names)
        ds = image.to_dataset(name='band_data')

        data_vars = {}
        for i, band_name in enumerate(band_names):
            band_data = ds['band_data'].isel(band=i).astype(np.int16)
            band_data = band_data.squeeze(drop=True).drop('band')
            data_vars[band_name] = band_data

        ds = xr.Dataset(data_vars)
        ds.attrs['time'] = str(np.datetime64(self.date).astype('datetime64[ns]'))

        return ds

    def stack_bands(self, resolution=10, reset=False):
        # maybe implement the option to create a dask delayed object for the whole process and then compute it outside this class
        try:
            if reset:
                self.reset()
            if not self.path.exists():
                band_names, resampled_bands = self.resample_bands(resolution)
                ds = self.create_dataset(band_names, resampled_bands)
                ds.rio.to_raster(self.path)
            else:
                print(f"File already exists: {self.path}")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        if self.path.exists():
            with rxr.open_rasterio(self.path) as src:
                return src
        else:
            return None
            
    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta
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
        self.orbit_state = self._initialize_orbit_state()
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
    
    def _initialize_orbit_state(self):
        orbit_names = {"asc": "ascending", "dsc": "descending"}
        orbit_info = Path(next(iter(self.band_files.values()))).stem.split("_")[-1]
        return orbit_names[orbit_info]

    def stack_bands(self, resolution=10, reset=False):
        try:
            if reset and self.path.exists():
                self.path.unlink()
                print(f"Removed existing file: {self.path}")

            if not self.path.exists():
                if len(self.band_files.values()) > 1:
                    band_names = []
                    resampled_bands = []
                    for band_name, band_path in self.band_files.items():
                        with rxr.open_rasterio(band_path) as raster:
                            if raster.rio.resolution() != (resolution, resolution):
                                raster = resample_raster(raster, resolution)
                            resampled_bands.append(raster)
                            band_names.append(band_name)

                    stacked_raster = xr.concat(resampled_bands, dim='band')
                    stacked_raster = stacked_raster.assign_coords(band=band_names)

                    ds = stacked_raster.to_dataset(name='band_data')
                    ds.attrs["time"] = np.datetime64(self.date).astype('datetime64[ns]')
                    
                    for i, band_name in enumerate(self.band_files.keys()):
                        ds[band_name] = ds['band_data'].isel(band=i)

                    ds = ds.drop_vars('band')
                    ds = ds.drop_vars('band_data')
                    ds.rio.to_raster(self.path) #ds.to_netcdf(self.path, compute=True)
                    print(f"Stacked bands successfully and saved as GeoTIFF under: {self.path}")
                else:
                    print(f"Only one band available. No need to stack. Renaming the file to {self.name}.tif")
                    for band_name, band_path in self.band_files.items():
                        band_path.rename(self.path)
            else:
                print(f"File already exists: {self.path}")
            return self
        
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                image = src.read()
            return image
        else:
            return None
            
    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta
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