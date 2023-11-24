import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
from .utils import patchify, band_management_decorator
from .band import SatelliteBand
from .annotation import Sentinel1Annotation, Sentinel2Annotation
from pathlib import Path

# TODO: reprojecting of patches is not working...they do not align with the original raster

# evtl add band_management where i resample and stack bands in advance? - but i only use it once so prob not relevant 

class SatelliteImage:
    def __init__(self):
        self._band_files_by_id = {}
        self._loaded_bands = {}
        self.annotation = None
        self.array = None
        self.date = None
        self.meta = {}

    def __getitem__(self, band_id):
        if band_id not in self._loaded_bands:
            band_path = self._band_files_by_id.get(band_id)
            if band_path:
                self._loaded_bands[band_id] = SatelliteBand(band_id, band_path)
            else:
                raise ValueError(f"Band path for '{band_id}' not found.")
        return self._loaded_bands[band_id]
    
    @band_management_decorator
    def _update_metadata(self):
        satellite_image_meta = {}
        for band_id, band in self._loaded_bands.items():
            satellite_image_meta[band_id] = band.meta
        self.meta = satellite_image_meta
        return self.meta     

    def add_band(self, band_id, file_path):
        """
        Add a new SatelliteBand object to the bands dictionary.
        """
        self._band_files_by_id[band_id] = file_path
        self._update_metadata()
        return

    def load_band(self, band_id):
        """
        Load and return a SatelliteBand object for the specified band.
        Args:
            band_id (str): The name of the band in uppercase letter to load.
        Returns:
            SatelliteBand: The loaded SatelliteBand object.
        """
        if band_id not in self._loaded_bands:
            band_path = self._band_files_by_id.get(band_id)
            if band_path:
                self._loaded_bands[band_id] = SatelliteBand(band_id, band_path)
            else:
                raise ValueError(f"Band path for '{band_id}' not found.")
        return self._loaded_bands[band_id]

    def unload_band(self, band_id):
        del self._loaded_bands[band_id]

    def load_all_bands(self):
        """
        Load all bands specified in the band_paths.
        This method iterates through all the band paths and loads each band.
        """
        for band_id in self._band_files_by_id.keys():
            self.load_band(band_id)
        return self._loaded_bands
    
    def unload_all_bands(self):
        self._loaded_bands = {}

    @band_management_decorator
    def resample(self, resolution):
        for band_id, band in self._loaded_bands.items():
            band = band.resample(resolution)
            self._band_files_by_id[band_id] = band.path
        self._update_metadata()
        return 
    
    @band_management_decorator
    def stack_bands(self):
        try:
            bands_stacked = np.vstack([band.array for band in self._loaded_bands.values()])
            self.array = bands_stacked
        except ValueError:
            raise ValueError(f"An error occurred while stacking bands: {ValueError}. Please make sure that all loaded bands are resampled to the same resolution.")
        return self.array
    
        # patches_array = np.stack(bands_patches, axis=1) # (1972,10,128,128)
        #     patches_meta = [] # 1972 meta items
        #     for band_meta in band_patches_meta:
        #         patches_array_meta = band_meta
        #         patches_array_meta["count"] = patches_array.shape[1]
        #         patches_meta.append(patches_array_meta)
        #     return patches_array, patches_meta # (1972,10,128,128), [meta1, meta2, .... meta1972]
        # except ValueError:
        #     raise ValueError(f"An error occurred while generating patches of all bands: {ValueError}. Please make sure that all bands are resampled to the same resolution.")

    @band_management_decorator
    def create_patches_with_metadata(self, patch_size):
        # Bedingung ist das stacken von bändern und das resamplen -> change this 
    
        try:
            bands_patches = []
            for band in self._loaded_bands.values():
                band_patches = band.create_patches(patch_size)
                band_patches.append(np.array(band_patches)) # (1972,1,128,128)
            satellite_image_patches = np.stack(bands_patches, axis=1) # (1972,10,128,128)
            satellite_image_patches_metadata = []
            band_patches_metadata = self.load_band("B02").get_patches_metadata(list(bands_patches)[0])
            for band_meta in band_patches_metadata:
                band_meta["count"] = satellite_image_patches.shape[1]
                satellite_image_patches_metadata.append(band_meta)
            return list(satellite_image_patches), satellite_image_patches_metadata # two list with same length 
        except ValueError:
            raise ValueError(f"An error occurred while creating patches of SatelliteImage: {ValueError}. Please make sure that bands are resampled. Use for this the function resample().") 

    def save_patches(self, patches, patches_metadata):
        patches_folder = self.path.parent / "patches"
        patches_folder.mkdir(parents=True, exist_ok=True)
        for idx, patch, patch_meta in enumerate(zip(patches, patches_metadata)):
            patch_meta['driver'] = 'GTiff'
            patch_path = patches_folder / (self.name + f"patch{idx}.tif" )
            with rasterio.open(patch_path, 'w', **patch_meta) as dst:
                dst.write(patch[0, :, :], 1)
        return patches_folder

    @band_management_decorator
    def calculate_spectral_signature(self, annotation_shapefile):
        annotation = SatelliteImageAnnotation(annotation_shapefile)
        geometries = annotation.get_geometries_as_list()
        spectral_sig = {}
        for band_id, band in self._loaded_bands.items():
            with rasterio.open(band.path, "r") as src:
                mean_values = [np.mean(mask(src, [polygon], crop=True)[0]) for polygon in geometries]
                spectral_sig[band_id] = np.mean(mean_values)
        return spectral_sig

class Sentinel2(SatelliteImage):
    
    def __init__(self, folder_of_satellite_image, date, annotation_shapefile=None):
        super().__init__()   
        self.base_folder = Path(folder_of_satellite_image)
        self._annotation_shapefile = annotation_shapefile
        self._band_files_by_id = self._initialize_bands_path()    
        self._scl = self._initialize_scl()
        self.meta = self._update_metadata()
        self.date = date
    
    def _initialize_bands_path(self):
        bands_path = {}
        file_paths = [entry for entry in self.base_folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                bands_path[band_id] = band_path
        return self._sort_band_paths(bands_path)
    
    def _initialize_scl(self):
        scl_file = list(self.base_folder.glob('*SCL.tif'))[0]
        self._scl = {"SCL": scl_file}
        return self._scl

    def _sort_band_paths(self, bands_path):
        sorted_keys = sorted(bands_path.keys(), key=self._extract_band_number) # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL
        sorted_bands_paths = {k: bands_path[k] for k in sorted_keys}
        return sorted_bands_paths

    def _extract_band_info(self, file_path):
        file_name = file_path.name  
        pattern = r"(B\d+[A-Z]?)\.tif"
        match = re.search(pattern, file_name)
        return match.group(1) if match else None
            
    def _extract_band_number(self, key):
        match = re.match(r"B(\d+)", key)
        return int(match.group(1)) if match else float('inf')  # if key does not start with 'B', place it at the end
    
    def _load_scl(self):
        return SatelliteBand("SCL", self._scl["SCL"])

    def _unload_scl(self):
        self._scl = {}

    def _load_annotation(self):
        if self._annotation_shapefile:
            self.annotation = Sentinel2Annotation(self, self._annotation_shapefile)
            return self.annotation

    def calculate_bad_pixels(self):
        scl_band = self._load_scl()
        bad_pixel_count = np.isin(scl_band.array, [0,1,2,3,8,9,10,11])
        self._unload_scl()
        return np.mean(bad_pixel_count) * 100
    
    def is_quality_acceptable(self, bad_pixel_ratio=15):
        return self.calculate_bad_pixels() < bad_pixel_ratio

    @band_management_decorator     
    def calculate_ndvi(self):
        red = self._loaded_bands["B04"].array
        nir = self._loaded_bands["B08"].array
        np.seterr(divide='ignore', invalid='ignore')
        return (nir.astype(float) - red.astype(float)) / (nir + red)
    
    @band_management_decorator
    def calculate_ndwi(self):
        nir = self._loaded_bands["B08"].array
        swir1 = self._loaded_bands["B11"].resample(10).array
        np.seterr(divide='ignore', invalid='ignore')
        return (nir.astype(float) -swir1.astype(float)) / (nir + swir1)

    def save_and_add_index(self, index_name, index):
        sample_band_meta = self.meta["B02"]
        profile = {
            'driver': 'GTiff',
            'height': index.shape[1],
            'width': index.shape[2],
            'count': 1,
            'dtype': index.dtype,
            'crs': sample_band_meta["crs"],
            'transform': sample_band_meta["transform"],
            'compress': 'lzw',
            'nodata': None
        }
        index_path = self.base_folder / f"S2_{self.date.strftime('%Y%m%d')}_{index_name}.tif"
        with rasterio.open(index_path, 'w', **profile) as dst:
            dst.write(index)
        self.add_band(index_name, index_path)
        return 

    @band_management_decorator
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
     
    def __init__(self, folder_of_satellite_image, date):
        super().__init__()   
        self.base_folder = Path(folder_of_satellite_image)
        self._band_files_by_id = self._initialize_bands_path()    
        self.meta = self._update_metadata()
        self.date = date
    
    def _initialize_bands_path(self):
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

    @band_management_decorator
    def calculate_cross_ratio(self):
        vv = self._loaded_bands["VV"]
        vh = self._loaded_bands["VH"]
        return vh / vv
    
    @band_management_decorator
    def calculate_polarization_difference(self):
        vv = self._loaded_bands["VV"]
        vh = self._loaded_bands["VH"]
        return np.nan_to_num(vv-vh)
    
    def save_and_add_index(self, index_name, index):
        sample_band_meta = self.meta["VV"]
        profile = {
            'driver': 'GTiff',
            'height': index.shape[1],
            'width': index.shape[2],
            'count': 1,
            'dtype': index.dtype,
            'crs': sample_band_meta["crs"],
            'transform': sample_band_meta["transform"],
            'compress': 'lzw',
            'nodata': None
        }
        index_path = self.base_folder / f"S1_{self.date.strftime('%Y%m%d')}_{index_name}.tif"
        with rasterio.open(index_path, 'w', **profile) as dst:
            dst.write(index)
        self.add_band(index_name, index_path)
        return 

    def plot_cross_ratio(self):
        try:
            cr = np.moveaxis(self.load_band("CR"),0,-1)
            plt.imshow(cr)
            plt.title(f"Cross ratio of sentinel-1 image on date: {self.date}")
            plt.show()
            plt.savefig("cr.png")
        except ValueError:
            raise ValueError(f"An error occurred while plotting cross ratio: {ValueError}. Please make sure that specific index is calcluated and added to bands.")


