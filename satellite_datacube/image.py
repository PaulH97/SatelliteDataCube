import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
import pandas as pd 
import geopandas as gpd
from datetime import datetime
from .utils import patchify, contrast_stretching
from .band import SatelliteBand
from rasterio.features import geometry_mask
from pathlib import Path

# TODO: reprojecting of patches is not working...they do not align with the original raster
# - do i need to unload bands in every function?

class Annotation:
    def __init__(self, shapefile_path):
        self.path = shapefile_path
        self.df = self._load_dataframe()
        self.band = None
    	
    def _load_dataframe(self):
        return gpd.read_file(self.path) 

    def _repair_geometries(self):
        
        geometries = self.df["geometry"].to_list()

        repaired_geometries = []
        for geom in geometries:
            if geom:
                if geom.is_valid:
                    repaired_geometries.append(geom)
                else:
                    repaired_geometries.append(geom.buffer(0))

        self.df["geometry"] = repaired_geometries
        return  
    
    def get_geometries_as_list(self):
        self._repair_geometries()
        return self.df["geometry"].to_list()

    def rasterize_annotation(self, raster_metadata):
                
        geometries = self.get_geometries_as_list()
        raster_metadata['dtype'] = 'uint8'
        
        output_path = self.path.parent / (self.path.stem + ".tif")
        with rasterio.open(output_path, 'w', **raster_metadata) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)

        self.band = SatelliteBand(band_name="annotation", band_path=output_path)
        return self.band

    def plot(self):
        return self.band.plot()


class SatelliteImage:
    def __init__(self, band_paths):
        """
        Initialize the SatelliteImage with paths to the satellite band data.

        Args:
            band_paths (dict): A dictionary mapping band names to their file paths.
            loaded_bands (dict): 
            array (numpy.darray):
            seed (int):
        """
        self.band_paths = band_paths 
        self.loaded_bands = {}
        self.seed = 42

    def __getattr__(self, attr_name):
        try:
            return self.load_band(attr_name.upper())
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")

    def add_band(self, band_id, file_path):
        """
        Add a new SatelliteBand object to the bands dictionary.
        """
        self.band_paths[band_id] = file_path

    def load_band(self, band_name):
        """
        Load and return a SatelliteBand object for the specified band.
        Args:
            band_name (str): The name of the band in uppercase letter to load.
        Returns:
            SatelliteBand: The loaded SatelliteBand object.
        """
        if band_name not in self.loaded_bands:
            band_path = self.band_paths.get(band_name)
            if band_path:
                self.loaded_bands[band_name] = SatelliteBand(band_name, band_path)
            else:
                raise ValueError(f"Band path for '{band_name}' not found.")
        return self.loaded_bands[band_name]

    def load_all_bands(self):
        """
        Load all bands specified in the band_paths.
        This method iterates through all the band paths and loads each band.
        """
        for band_name in self.band_paths:
            self.load_band(band_name)
        return
    
    def unload_all_bands(self):
        self.loaded_bands = {}

    def stack_resampled_bands(self):
        if not self.loaded_bands:
            self.load_all_bands()
        
        band_arrays = []
        for band_name, band in self.loaded_bands.items():
            band = band.resample(10)
            band_arrays.append(band.array)

        bands_stacked = np.vstack(band_arrays)  
        self.unload_all_bands()
        return bands_stacked 

    def create_patches(self, patch_size):
        stacked_bands = self.stack_resampled_bands()
        return patchify(stacked_bands, patch_size)  
 
    def calculate_spectral_signature(self, annotation):
        self.load_all_bands()
        geometries = annotation.get_geometries_as_list()
        spectral_sig = {}
        for band_id, band in self.loaded_bands.items():
            if band_id != "SCL":
                with rasterio.open(band.path, "r") as src:
                    mean_values = [np.mean(mask(src, [polygon], crop=True)[0]) for polygon in geometries]
                    spectral_sig[band_id] = np.mean(mean_values)
        return spectral_sig
       
    def plot_spectral_signature(self, spectral_signature):
        band_ids = list(spectral_signature.keys())
        reflectances = list(spectral_signature.values())

        plt.figure(figsize=(10, 5))
        plt.plot(band_ids, reflectances, marker='o', linestyle='-')
        plt.title("Spectral Signature")
        plt.xlabel("Band ID")
        plt.ylabel("Reflectance")
        plt.grid(True)
        plt.show()

class Sentinel2(SatelliteImage):
    
    def __init__(self, si_folder, location, date):
        self.folder = si_folder
        self.location = location
        self.date = date
        bands_path = self._construct_bands_path()
        super().__init__(bands_path)

    def _construct_bands_path(self):
        band_paths = {}
        file_paths = [entry for entry in self.folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                band_paths[band_id] = band_path
        return self._sort_band_paths(band_paths)

    def _sort_band_paths(self, band_paths):
        sorted_keys = sorted(band_paths.keys(), key=self._extract_band_number) # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL
        sorted_bands_paths = {k: self.bands[k] for k in sorted_keys}
        return sorted_bands_paths

    def _extract_band_info(self, file_path): 
        pattern = r"(B\d+[A-Z]?|SCL)\.tif"
        match = re.search(pattern, file_path)
        return match.group(1) if match else None
            
    def _extract_band_number(self, key):
        match = re.match(r"B(\d+)", key)
        return int(match.group(1)) if match else float('inf')  # if key does not start with 'B', place it at the end
    
    def calculate_bad_pixels(self):
        scene_class = self.load_band("SCL")
        bad_pixel_count = np.isin(scene_class.array, [0,1,2,3,8,9,10,11])
        self.unload_all_bands()
        return np.mean(bad_pixel_count) * 100

    def calculate_ndvi(self):
        red = self.load_band("B04").array
        nir = self.load_band("B08").array
        np.seterr(divide='ignore', invalid='ignore')
        self.unload_all_bands()
        return (nir.astype(float) - red.astype(float)) / (nir + red)
    
    def calculate_ndvi(self):
        nir = self.load_band("B08").array
        swir1 = self.load_band("B11").resample(10).array
        np.seterr(divide='ignore', invalid='ignore')
        self.unload_all_bands()
        return (nir.astype(float) -swir1.astype(float)) / (nir + swir1)

    def save_index(self, index_name, index):
        b02 = self.load_band("B02")
        profile = {
            'driver': 'GTiff',
            'height': index.shape[1],
            'width': index.shape[2],
            'count': 1,
            'dtype': index.dtype,
            'crs': b02.meta["crs"],
            'transform': b02.meta["transform"],
            'compress': 'lzw',
            'nodata': None
        }
        index_path = self.folder / f"S2_{self.date.strftime('%Y%m%d')}_{index_name}.tif"
        with rasterio.open(index_path, 'w', **profile) as dst:
            dst.write(index)
        self.add_band(index_name, index_path)
        self.unload_all_bands()
        return 

    def plot_rgb(self):
        red = self.load_band("B04").stretch_contrast()
        green = self.load_band("B03").stretch_contrast()
        blue = self.load_band("B02").stretch_contrast()
        rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
        plt.imshow(rgb)
        plt.title(f"RGB of {self.location} on date: {self.date}")
        plt.show()
        plt.savefig("rgb.png")
        return        
        
class Sentinel1(SatelliteImage):
    
    def __init__(self, si_folder, location, date):
        super().__init__()
        self.folder = si_folder 
        self.location = location
        self.date = date

    def calculate_indices(self, out_dir, path=True):
        vv = self.read_band_3D("VV")
        vh = self.read_band_3D("VH")
        cr = np.nan_to_num(vh-vv)

        # Get affine transformation and CRS from the original raster (e.g., "B4")
        with rasterio.open(self.get_band_path("VV")) as src:
            transform = src.transform
            crs = src.crs

        # Define rasterio profile for the output files
        profile = {
            'driver': 'GTiff',
            'height': cr.shape[0],
            'width': cr.shape[1],
            'count': 1,
            'dtype': cr.dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': None
        }

        # Save NDVI and NDWI files with the affine transformation and CRS
        cr_path = os.path.join(out_dir, f"{self.tile_id}_CR.tif")
       
        with rasterio.open(cr_path, 'w', **profile) as dst:
            dst.write(cr[:,:,0], 1)

        self.indizes_path = [cr_path]
        print("Calculated Sentinel-1 indizes")
    
        if path:
            return cr_path
        else: 
            return 

    def plot_rgb(self):

        def contrastStreching(image):
            
            image = image.astype(np.float32)
            csImg = np.empty_like(image, dtype=np.float32)

            # Perform contrast stretching on each channel
            for band in range(image.shape[-1]):
                imgMin = image[...,band].min().astype(np.float32)
                imgMax = image[...,band].max().astype(np.float32)
                csImg[...,band] = (image[...,band] - imgMin) / (imgMax - imgMin)
            
            return csImg

        if self.bands:
            
            red = self.get_band("B04").normalize()
            green = self.get_band("B03").normalize()
            blue = self.get_band("B02").normalize()
            
            rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
            rgb = contrastStreching(rgb)
            plt.figure(figsize=(10,10))
            plt.imshow(rgb)
            plt.show()

        else: 
            raise AttributeError(f"No rgb image is initiated. Please initiate the corresponding rgb image using the initiate_bands() method and try again.")

        return

    def plot_backscatter():
        return
    
class Sentinel12(Sentinel1, Sentinel2):
    def __init__(self, si_folder, location, date):
        super().__init__()
        self.s1 = Sentinel1(si_folder, location, date)
        self.s2 = Sentinel2(si_folder, location, date)
        self.indizes_path = []

    def initiate_bands(self):
        
        self.s1.initiate_bands()
        self.s2.initiate_bands()
        
        return 
        
    def calculate_indices(self, out_dir, path=True):
        
        s1_idx = self.s1.calculate_indizes(out_dir)
        s2_idx = self.s2.calculate_indizes(out_dir)
        
        # Combine the idx as list
        combined_idx = [s1_idx] + s2_idx
        combined_idx.sort()
        self.indizes_path = combined_idx
        
        print("Calculated Sentinel-1/-2 indizes")
    
        if path:
            return combined_idx
        else: 
            return 
    
    def plot_rgb(self):

        def contrastStreching(image):
            
            image = image.astype(np.float32)
            csImg = np.empty_like(image, dtype=np.float32)

            # Perform contrast stretching on each channel
            for band in range(image.shape[-1]):
                imgMin = image[...,band].min().astype(np.float32)
                imgMax = image[...,band].max().astype(np.float32)
                csImg[...,band] = (image[...,band] - imgMin) / (imgMax - imgMin)
            
            return csImg

        if self.bands:
            
            red = self.get_band("B04").normalize()
            green = self.get_band("B03").normalize()
            blue = self.get_band("B02").normalize()
            
            rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
            rgb = contrastStreching(rgb)
            plt.figure(figsize=(10,10))
            plt.imshow(rgb)
            plt.show()

        else: 
            raise AttributeError(f"No rgb image is initiated. Please initiate the corresponding rgb image using the initiate_bands() method and try again.")

        return
    
    def plot_backscatter():
        return
