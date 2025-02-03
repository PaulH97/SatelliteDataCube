from pathlib import Path
from datetime import datetime
from satellite_datacube.utils_light import extract_S2_band_name, extract_S1_band_name, resample_raster, extract_band_number, load_file, normalize
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import rasterio
import numpy as np
import gc
import re
from rasterio.warp import reproject, Resampling as RioResampling

class Sentinel1():
    def __init__(self, folder):
        self.folder = Path(folder)
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.orbit_state = self._initialize_orbit_state()
        self.path = self.folder / f"{self.name}.tif"

    def _initialize_name(self):
        raster_filepath = Path(next(iter(self.band_files.values())))
        name_parts = raster_filepath.stem.split("_")
        return "_".join(name_parts[:-2])

    def _initialize_date(self):
        match = re.search(r"\d{8}", self.name)  # Updated to search for 8-digit date format
        if match:
            date = datetime.strptime(match.group(0), "%Y%m%d").date()  # Updated format to YYYYMMDD
            return date
        else:
            raise ValueError(f"Date string does not match the expected pattern.")

    def _initialize_bands(self):
        bands = {}
        for tif_file in self.folder.glob('*.tif'):
            band_name = extract_S1_band_name(tif_file)
            if band_name:
                band_name = band_name.upper()
                bands[band_name] = tif_file
        sorted_keys = sorted(bands.keys())
        bands = {k: bands[k] for k in sorted_keys}
        return bands
    
    def _initialize_orbit_state(self):
        orbit_names = {"asc": "ascending", "dsc": "descending"}
        for tif_file in self.band_files.values():
            if 'asc' in tif_file.stem.lower():
                return orbit_names['asc']
            elif 'dsc' in tif_file.stem.lower():
                return orbit_names['dsc']
        raise ValueError("No valid orbit state found in the file names.")
                
    def stack_bands(self):
            try:
                # Use the transform and CRS of the first band as the reference
                first_band_path = next(iter(self.band_files.values()))
                with rasterio.open(first_band_path) as ref:
                    ref_transform = ref.transform
                    ref_crs = ref.crs
                    ref_width = ref.width
                    ref_height = ref.height

                stacked_bands = []
                band_names = []

                for band_name, band_path in self.band_files.items():
                    with rasterio.open(band_path) as src:
                        if src.transform != ref_transform or src.crs != ref_crs:
                            # Reproject the raster to match the reference transform and CRS
                            data = src.read(1)
                            reprojected_data = np.empty((ref_height, ref_width), dtype=src.dtypes[0])
                            reproject(
                                source=data,
                                destination=reprojected_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref_transform,
                                dst_crs=ref_crs,
                                resampling=RioResampling.bilinear
                            )
                        else:
                            reprojected_data = src.read(1)

                        stacked_bands.append(reprojected_data)
                        if band_name:
                            band_names.append(band_name)

                # Save the stacked bands as a GeoTIFF with band descriptions
                with rasterio.open(
                    self.path,
                    "w",
                    driver="GTiff",
                    height=ref_height,
                    width=ref_width,
                    count=len(stacked_bands),
                    dtype=stacked_bands[0].dtype,
                    crs=ref_crs,
                    transform=ref_transform
                ) as dst:
                    for idx, (band_data, band_name) in enumerate(zip(stacked_bands, band_names), start=1):
                        dst.write(band_data, idx)
                    # Set all band descriptions at once:
                    dst.descriptions = tuple(band_names)

            except Exception as e:
                print(f"An error occurred: {e}")
                raise
            
    def load_data(self):
        if self.path.exists():
            return load_file(self.path)
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
        if "SCL" not in bands.keys():
            print(f"SCL missing for: {self.folder}")
        return bands

    def stack_bands(self):
        try:
            # Use the transform and CRS of the first band as the reference
            first_band_path = next(iter(self.band_files.values()))
            with rasterio.open(first_band_path) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_width = ref.width
                ref_height = ref.height

            stacked_bands = []
            band_names = []

            for band_name, band_path in self.band_files.items():
                with rasterio.open(band_path) as src:
                    if src.transform != ref_transform or src.crs != ref_crs:
                        # Reproject the raster to match the reference transform and CRS
                        data = src.read(1)
                        reprojected_data = np.empty((ref_height, ref_width), dtype=src.dtypes[0])
                        reproject(
                            source=data,
                            destination=reprojected_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=RioResampling.bilinear
                        )
                    else:
                        reprojected_data = src.read(1)

                    stacked_bands.append(reprojected_data)
                    if band_name:
                        band_names.append(band_name)

            # Save the stacked bands as a GeoTIFF with band descriptions
            with rasterio.open(
                self.path,
                "w",
                driver="GTiff",
                height=ref_height,
                width=ref_width,
                count=len(stacked_bands),
                dtype=stacked_bands[0].dtype,
                crs=ref_crs,
                transform=ref_transform
            ) as dst:
                for idx, (band_data, band_name) in enumerate(zip(stacked_bands, band_names), start=1):
                    dst.write(band_data, idx)
                # Set all band descriptions at once:
                dst.descriptions = tuple(band_names)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        if self.path.exists():
            return load_file(self.path)
        else:
            return None
                    
    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta
            return meta
        else:
            return None
        
    def calculate_bad_pixel_ratio(self):
        data = self.load_data()
        if data is None:
            raise ValueError(f"No data available for {self.name}. Please ensure the bands are stacked.")
        if 'SCL' not in data:
            raise ValueError("SCL band is missing from the dataset. Cannot calculate bad pixel ratio.")
        scl_band = data['SCL'].values  
        bad_pixels = np.isin(scl_band, [0, 1, 2, 3, 8, 9, 10, 11])
        bad_pixel_ratio = np.sum(bad_pixels) / scl_band.size * 100
        return bad_pixel_ratio
    
    def get_ndvi_path(self):
        ndvi_path = self.path.parent / f"{self.name}_NDVI.tif"
        if not ndvi_path.exists():
            self.calculate_ndvi(save=True)
        return ndvi_path
    
    def calculate_ndvi(self, save=False): 
        # Early exit if the input path does not exist
        if not self.path.exists():
            raise FileNotFoundError(f"The path {self.path} does not exist.")
        
        image = self.load_data()
        nir = np.clip(image.B08.values, 0, 10000)
        red = np.clip(image.B04.values, 0, 10000)

        # Scale to 0-1 range
        nir = nir / 10000.0
        red = red / 10000.0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
        
        ndvi = np.nan_to_num(ndvi, nan=np.nan, posinf=np.nan, neginf=np.nan)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi_data_array = xr.DataArray(
            ndvi,
            dims=['time', 'y', 'x'],  
            coords={
                'time': image['time'].data,
                'y': image['y'].data,
                'x': image['x'].data
            },
            name='NDVI'
        )
        
        image = image.assign(NDVI=ndvi_data_array)
        
        if save:
            ndvi_path = self.path.parent / f"{self.name}_NDVI.tif"
            if ndvi_path.exists():
                ndvi_path.unlink()
            image['NDVI'].rio.to_raster(ndvi_path) 
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

