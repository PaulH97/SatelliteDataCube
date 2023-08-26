import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
import pandas as pd 
import fiona
from datetime import datetime
from utils import patchify, save_patch
from band import SatelliteBand

class SatelliteImage:
    def __init__(self):
        self._bands = {} # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL'
        self._mask = None
        self._stackedBands = None
        self._badPixelRatio = None
        self._seed = 42

    def __getattr__(self, attr_name):
        try:
            return self.get_band(attr_name.upper())
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")
        
    def _extract_band_info(self, file_path): 
        pattern = r"(B\d+[A-Z]?|SCL)\.tif"
        match = re.search(pattern, file_path)
        return match.group(1) if match else None
            
    def _extract_band_number(self, key):
        match = re.match(r"B(\d+)", key)
        return int(match.group(1)) if match else float('inf')  # if key does not start with 'B', place it at the end
    
    def get_band(self, band_id):
        if not self._bands:
            self.initiate_bands
        band = self._bands.get(band_id)
        if band is None:
            raise AttributeError(f"No band with id '{band_id}' found")
        return band
    
    def add_band(self, band_id, file_path):
        self._bands[band_id] = SatelliteBand(band_id, file_path)

    def initiate_attributes(self, indices=False):
        self.initiate_bands(indices) 
        self.initiate_mask()
        self.stack_bands()
        self.calculate_bad_pixels()
        return
    
    def initiate_bands(self, indices=False):
        bands = {}
        bands_path = [entry.path for entry in os.scandir(self.folder) if entry.is_file() and entry.name.endswith('.tif')]
        for band_path in bands_path:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                self.add_band(band_id, band_path)
        
        sorted_keys = sorted(self._bands, key=self._extract_band_number)
        self._bands = {k: self._bands[k] for k in sorted_keys}
        if indices:
            self.calculate_indices()
        return

    def initiate_mask(self):
        bands_path = [entry.path for entry in os.scandir(self.folder) if entry.is_file() and entry.name.endswith('.tif')]
        available_mask = False
        for band_path in bands_path:
            if "annotation_10m" in os.path.basename(band_path):
                available_mask = True
                mask = rasterio.open(band_path).read()  
        if not available_mask: 
            src_metadata = rasterio.open(bands_path[0]).meta
            mask = np.zeros((1, src_metadata['height'], src_metadata['width']), dtype=src_metadata['dtype'])
        self._mask = mask
        return
    
    def unload_bands(self):
        self._bands = {}

    def unload_mask(self):
        self._mask = None
            
    def check_dir_content(self, dir_path):
        if os.path.isdir(dir_path):
            # List all files and subdirectories in the given directory           
            dir_content = os.listdir(dir_path)
            if not dir_content:
                return None
            else:
                return dir_path
        else:
            return None
        
    def calculate_bad_pixels(self):
        if not self._bands:
            self.initiate_bands()
        scene_class = self._bands["SCL"].bandArray
        bad_pixel = np.isin(scene_class, [0,1,2,3,8,9,10,11])
        self._badPixelRatio = np.mean(bad_pixel) * 100
        return self._badPixelRatio
    
    def stack_bands(self, indices=False):
        if not self._bands:
            self.initiate_bands(indices=indices)
        resampled_bands = [band.resample(resolution=10, reference_band_path=self._bands["B02"].path) for band in self._bands.values()]
        self._stackedBands = np.vstack(resampled_bands) # ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', ('NDVI', 'NDWI'), 'SCL'] 
        return self._stackedBands
        
    def process_patches(self, patch_size, source, indices=False, save=False, output_folder=None):
        if source == "img":
            self.stack_bands(indices=indices)
            stackedBands = np.delete(self._stackedBands, -1, axis=0)
            patches = patchify(stackedBands, patch_size)
        elif source == "msk":
            if not self._mask:
                self.initiate_mask()
            patches = patchify(self._mask, patch_size)
            patches = [np.where(patch >= 1, 1, patch) for patch in patches]
        if save and patches and output_folder:
            band = next(iter(self._bands.values()))
            with rasterio.open(band.path) as src:
                template_meta = src.meta
            for i, patch in enumerate(patches):
                x, y = (i // (self._stackedBands.shape[1] // patch_size)) * patch_size, (i % (self._stackedBands.shape[1] // patch_size)) * patch_size
                save_patch(output_folder, x, y, patch, template_meta, patch_size, source)
        return patches

    def load_patches_asPath(self, patches_folder):
        patches_folder = [os.path.join(patches_folder, file) for file in os.listdir(patches_folder) if file == "patches"][0]
        patches = {}
        for source in os.listdir(patches_folder):
            sourceFolder = os.path.join(patches_folder, source)
            patches[source] = [os.path.join(sourceFolder, patch) for patch in os.listdir(sourceFolder)]
        return patches
 
    def calculate_spectral_signature(self, shapefile, csv_file=False):

        if not self._stackedBands:
            self.stack_bands()

        with fiona.open(shapefile, "r") as file:
            geometries = [feature["geometry"] for feature in file]

        spectral_sig = {}
        for band_id, band in self._bands.items():
            if band_id != "SCL":
                band = band.band # open rasterio dataset
                mean_values = [np.mean(mask(band, [polygon], crop=True)[0]) for polygon in geometries]
                print(mean_values)
                spectral_sig[band_id] = np.mean(mean_values)

        if csv_file:
            spectral_sig_df = pd.DataFrame(spectral_sig)
            spectral_sig_df.to_csv("spectral_signature.csv")
            return spectral_sig_df
        else:
            return spectral_sig
       
    def plot_spectral_signature(self, shapefile):
        """
        Plot the spectral signature for a given region.
        """
        spectral_signature = self.calculate_spectral_signature(shapefile)
        print(spectral_signature)
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
    
    def __init__(self, si_folder):
        super().__init__()
        self.folder = si_folder #
        self.location = os.path.basename(os.path.dirname(self.folder))
        self.date = datetime.strptime(os.path.basename(self.folder), "%Y%m%d")

    def calculate_indices(self, save_file=True):
        
        exampleRaster = self.get_band("B02")
        red = self.get_band("B04").bandArray
        nir = self.get_band("B08").bandArray
        swir1 = self.get_band("B11").resample(resolution=10, reference_band_path=exampleRaster.path)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
        ndwi = (nir.astype(float) -swir1.astype(float)) / (nir + swir1)
        
        if save_file:
            profile = {
                'driver': 'GTiff',
                'height': ndvi.shape[1],
                'width': ndvi.shape[2],
                'count': 1,
                'dtype': ndvi.dtype,
                'crs': exampleRaster.band.crs,
                'transform': exampleRaster.band.transform,
                'compress': 'lzw',
                'nodata': None
            }

            ndvi_path = os.path.join(self.folder, "NDVI.tif")
            ndwi_path = os.path.join(self.folder, "NDWI.tif")

            with rasterio.open(ndvi_path, 'w', **profile) as dst:
                dst.write(ndvi)

            with rasterio.open(ndwi_path, 'w', **profile) as dst:
                dst.write(ndwi)

            self.add_band("NDVI", ndvi_path)
            self.add_band("NDWI", ndwi_path)
            keys_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL']
            self._bands = {key: self._bands[key] for key in keys_order}
            return self._bands
        else:
            return ndvi, ndwi

    def plot_rgb(self):
        
        def contrastStreching(image, lower_percentile=2, upper_percentile=98):
            image = image.astype(np.float32)
            csImg = np.empty_like(image, dtype=np.float32)
            for band in range(image.shape[-1]):
                imgMin = np.percentile(image[...,band], lower_percentile).astype(np.float32)
                imgMax = np.percentile(image[...,band], upper_percentile).astype(np.float32)
                csImg[...,band] = (image[...,band] - imgMin) / (imgMax - imgMin)
            return csImg

        if not self._bands:
            self.initiate_bands()

        red = self.get_band("B04").normalize()
        green = self.get_band("B03").normalize()
        blue = self.get_band("B02").normalize()
        rgb = np.moveaxis(np.vstack((red, green, blue)), 0, -1)
        rgb = contrastStreching(rgb)
        plt.figure(figsize=(10,10))
        plt.imshow(rgb)
        plt.show()
        return
   
class Sentinel1(SatelliteImage):
    
    def __init__(self, tile_id, month, year):
        super().__init__(tile_id, month, year)
        self.satellite = "S1"
        self.indizes_path = []
        self.tile_folder = f"" 

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

        if self._bands:
            
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
    def __init__(self, tile_id, month, year):
        super().__init__(tile_id, month, year)
        self.s1 = Sentinel1(tile_id, month, year)
        self.s2 = Sentinel2(tile_id, month, year)
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

        if self._bands:
            
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
