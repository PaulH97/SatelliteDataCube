import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
from .utils import patchify
from .band import SatelliteBand
from .annotation import SatelliteImageAnnotation
from pathlib import Path

# TODO: reprojecting of patches is not working...they do not align with the original raster
# - do i need to unload bands in every function?

class SatelliteImage:
    def __init__(self):
        """
        Initialize the SatelliteImage with paths to the satellite band data.

        Args:
            bands_path (dict): A dictionary mapping band names to their file paths.
            loaded_bands (dict): 
            array (numpy.darray):
            seed (int):
        """
        self._band_files_by_id = {}
        self.bands = {}
        self.annotation = None
        self.meta = {}

    def __getitem__(self, band_id):
        if band_id not in self.bands:
            band_path = self._band_files_by_id.get(band_id)
            self.bands[band_id] = SatelliteBand(band_path)  # Band class handles the loading of the array
        return self.bands[band_id]

    def _update_metadata(self):
        satellite_image_meta = {}
        for band_id in self._band_files_by_id.keys():
            satellite_image_meta[band_id] = self[band_id].meta
        self.meta = satellite_image_meta
        return      

    def add_band(self, band_id, file_path):
        """
        Add a new SatelliteBand object to the bands dictionary.
        """
        self._band_files_by_id[band_id] = file_path
        self._update_metadata()
        return
    
    def clear_bands(self):
        self.bands = {}

    def load_annotation(self, annotation_shapefile):
        self.annotation = SatelliteImageAnnotation(annotation_shapefile)
        return self.annotation

    def stack_and_resample_bands(self):
        if not self.loaded_bands_by_id:
            self.load_all_bands()
        band_arrays = []
        for band in self.loaded_bands_by_id.values():
            band = band.resample(10)
            band_arrays.append(band.array)
        bands_stacked = np.vstack(band_arrays)
        self.unload_all_bands()
        return bands_stacked 
    
    def create_patches(self, patch_size):
        stacked_bands = self.stack_and_resample_bands()
        return patchify(stacked_bands, patch_size)  
 
    def calculate_spectral_signature(self, annotation):
        self.load_all_bands()
        geometries = annotation.get_geometries_as_list()
        spectral_sig = {}
        for band_id, band in self.loaded_bands_by_id.items():
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
        plt.savefig("Spectral_sig.png")

class Sentinel2(SatelliteImage):
    
    def __init__(self, folder_of_satellite_image, date):
        super().__init__()
        self.base_folder = Path(folder_of_satellite_image)
        self._band_files_by_id = self._construct_bands_path()
        self.meta = self._update_metadata()
        self.date = date
        self.bands = {}              
    
    def _construct_bands_path(self):
        bands_path = {}
        file_paths = [entry for entry in self.base_folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                bands_path[band_id] = band_path
        return self._sort_band_paths(bands_path)

    def _sort_band_paths(self, bands_path):
        sorted_keys = sorted(bands_path.keys(), key=self._extract_band_number) # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL
        sorted_bands_paths = {k: bands_path[k] for k in sorted_keys}
        return sorted_bands_paths

    def _extract_band_info(self, file_path):
        file_name = file_path.name  
        pattern = r"(B\d+[A-Z]?|SCL)\.tif"
        match = re.search(pattern, file_name)
        return match.group(1) if match else None
            
    def _extract_band_number(self, key):
        match = re.match(r"B(\d+)", key)
        return int(match.group(1)) if match else float('inf')  # if key does not start with 'B', place it at the end
    
    def calculate_bad_pixels(self):
        scene_class = self["SCL"]
        bad_pixel_count = np.isin(scene_class.array, [0,1,2,3,8,9,10,11])
        return np.mean(bad_pixel_count) * 100
    
    def is_quality_acceptable(self, bad_pixel_ratio=15):
        return self.calculate_bad_pixels() < bad_pixel_ratio
         
    def calculate_ndvi(self):
        red = self["B04"].array
        nir = self["B08"].array
        np.seterr(divide='ignore', invalid='ignore')
        return (nir.astype(float) - red.astype(float)) / (nir + red)
    
    def calculate_ndwi(self):
        nir = self.load_band("B08").array
        swir1 = self.load_band("B11").resample(10).array
        np.seterr(divide='ignore', invalid='ignore')
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
        index_path = self.base_folder / f"S2_{self.date.strftime('%Y%m%d')}_{index_name}.tif"
        with rasterio.open(index_path, 'w', **profile) as dst:
            dst.write(index)
        self.add_band(index_name, index_path)
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
        self.base_folder = si_folder
        self.location = location
        self.date = date
        bands_path = self._construct_bands_path()
        super().__init__(bands_path)

    def _construct_bands_path(self):
        bands_path = {}
        file_paths = [entry for entry in self.base_folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                bands_path[band_id] = band_path
        return bands_path

    def _extract_band_info(self, file_path): 
        pattern = r"(VV|VH)"
        match = re.search(pattern, file_path)
        return match.group(1) if match else None

    def calculate_cross_ratio(self):
        vv = self.load_band("VV")
        vh = self.load_band("VH")
        return vh / vv
    
    def calculate_polarization_difference(self):
        vv = self.load_band("VV")
        vh = self.load_band("VH")
        return np.nan_to_num(vv-vh)
    
    def save_index(self, index_name, index):
        vv = self.load_band("VV")
        profile = {
            'driver': 'GTiff',
            'height': index.shape[1],
            'width': index.shape[2],
            'count': 1,
            'dtype': index.dtype,
            'crs': vv.meta["crs"],
            'transform': vv.meta["transform"],
            'compress': 'lzw',
            'nodata': None
        }
        index_path = self.base_folder / f"S1_{self.date.strftime('%Y%m%d')}_{index_name}.tif"
        with rasterio.open(index_path, 'w', **profile) as dst:
            dst.write(index)
        self.add_band(index_name, index_path)
        self.unload_all_bands()
        return 
    
    def plot_cross_ratio(self):
        if "CR" not in self.loaded_bands_by_id:
            cr = self.calculate_cross_ratio()
            self.save_index("CR", cr)
        cr = self.load_band("CR") # as SatelliteBand
        return cr.plot()
