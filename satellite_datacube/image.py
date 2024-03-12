import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
from rasterio.windows import Window
from .utils import band_management_decorator, extract_band_data_for_annotation, extract_nearby_ndvi_data, pad_patch, get_metadata_of_window
from .band import SatelliteBand
from .annotation import SatelliteImageAnnotation
from pathlib import Path
import json
from datetime import datetime

# TODO After removing self.scl_path -> fix methods where it was used  

def bands_required(f):
    """Decorator to ensure bands are initialized before calling the method."""
    def wrapper(self, *args, **kwargs):
        if not self.bands:
            raise ValueError("Bands have not been initialized.")
        return f(self, *args, **kwargs)
    return wrapper

class SatelliteImage:
    def __init__(self):
        self.folder = None # folder of all files correlated to the image like bands, patches etc. 
        self.name = None # S2_mspc_l2a_20190509_B02 - function that extract this nam
        self.path = None
        self.bands = {} # dict with band_ids, scl, mask and filepaths as value - for access single raster files
        self.indizes = {}
        self.date = None # date of image
        self.meta = None # metadata of image with all bands

    def is_stacked(self, indizes=False):
        if not self.path.exists() or self.path.suffix.lower() not in ['.tif', '.tiff']:
            return False
        try:
            with rasterio.open(self.path) as src:
                # Checking if there's at least one band with content
                if indizes:
                    if src.count == 13:  # src.count returns the number of bands in the dataset
                        return True
                    else:
                        return False
                else:
                    if src.count == 11:
                        return True
                    else:
                        return False                 
        except rasterio.errors.RasterioIOError:
            return False

    @bands_required
    def __getitem__(self, band_id):
        try:
            raster_filepath = self.bands[band_id]
            if raster_filepath:
                return SatelliteBand(band_id, raster_filepath)
        except KeyError:
            raise ValueError(f"Band id {band_id} not found.")
            
    @bands_required
    def find_band_by_res(self, target_res):
        for band_id in self.bands:
            band = self[band_id]  # Use __getitem__ to get the SatelliteBand instance
            resolution = band.get_resolution() # (10,10)
            if resolution[0] == target_res:
                return band
        return None

    @bands_required
    def stack_bands(self, resolution=10, indizes=False): 
        if not self.is_stacked(indizes=indizes):
            if indizes and not self.indizes:
                self.calculate_ndvi() # created index and store information in self.indizes
                self.calculate_ndwi()
            band_with_desired_res = self.find_band_by_res(resolution)
            bands_to_stack = self.bands.update(self.indizes) # update existing bands with indizes - (if indizes are selected) 
            with rasterio.open(band_with_desired_res.path) as src:
                meta = src.meta.copy()
                meta['count'] = len(bands_to_stack)  # Update the metadata to reflect the total number of bands   
            with rasterio.open(self.path, 'w', **meta) as dst:
                band_descriptions = []
                for i, (band_id, band_filepath) in enumerate(bands_to_stack.items(), start=1):
                    # SatelliteBand.resample modifies the band in place and updates its .array attribute
                    band = SatelliteBand(band_id, band_filepath).resample(resolution)
                    band_arr = np.squeeze(band.array) # drops first axis: (1,7336,4302) -> (7336,4302)
                    dst.write(band_arr, i)
                    band_descriptions.append(band_id)
                dst.descriptions = tuple(band_descriptions)
            self.meta = meta
        return self

    def create_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        with rasterio.open(self.path) as src:
            step_size = patch_size - overlay
            for i in range(0, src.width,  step_size):
                for j in range(0, src.height, step_size):
                    window = Window(j, i, patch_size, patch_size)
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
        super().__init__()   
        self.folder = Path(folder)
        self.bands = self._initialize_rasters() # bands (+ indizes) + scene layer
        self.indizes = self._init_indizes()
        self.name = self._initialize_name() # S2_mspc_l2a_20190509
        self.date = datetime.strptime(self.name.split("_")[-1], "%Y%m%d").date() # date(2019,05,09)
        self.path = self.folder / f"{self.name}.tif"
        
    def _initialize_name(self):
        raster_filepath = next(iter(self.bands.values()))
        band_name = raster_filepath.stem
        image_name = "_".join(band_name.split("_")[:-1]) 
        return image_name
    
    def _initialize_rasters(self):
        raster_files_dir = {}
        file_paths = [entry for entry in self.folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for raster_filepath in file_paths:
            band_id = self._extract_band_info(raster_filepath)
            if band_id:       
                raster_files_dir[band_id] = raster_filepath
        return self._sort_raster_filepaths(raster_files_dir)
    
    def _init_indizes(self):
        indizes = {}
        file_paths = [entry for entry in self.folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for raster_filepath in file_paths:
            if "NDVI" in raster_filepath.name:
                indizes["NDVI"] = raster_filepath
            elif "NDWI" in raster_filepath.name:
                indizes["NDWI"] = raster_filepath
            else:
                continue
        return indizes
    
    def _sort_raster_filepaths(self, raster_files_dir):
        sorted_keys = sorted(raster_files_dir.keys(), key=self._extract_band_number) # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'SCL'
        sorted_raster_files_dirs = {k: raster_files_dir[k] for k in sorted_keys}
        return sorted_raster_files_dirs
    
    @staticmethod
    def _extract_band_info(file_path):
        file_name = file_path.name  
        pattern = r"(B\d+[A-Z]?|SCL)\.tif"
        match = re.search(pattern, file_name)
        return match.group(1) if match else None
    
    @staticmethod        
    def _extract_band_number(key):
        if key == "SCL":
            return 100  
        elif key == "MSK":
            return 101  
        match = re.match(r"B(\d+)", key)
        return int(match.group(1)) if match else float('inf')  # if key does not start with 'B', place it at the end
    
    def calculate_bad_pixel_ratio(self):
        if self.scl_path:
            with rasterio.open(self.scl_path) as src:
                slc = src.read()
            bad_pixel_count = np.isin(slc, [0,1,2,3,8,9,10,11])
            self.bad_pixel_ratio = np.mean(bad_pixel_count) * 100
            return self.bad_pixel_ratio

    def is_quality_acceptable(self, bad_pixel_ratio=15):
        if not self.bad_pixel_ratio:
            self.calculate_bad_pixel_ratio()
        return self.bad_pixel_ratio < bad_pixel_ratio

    def calculate_scl_data_of_annotation(self):
        annotation_scl = {}
        annotation_df = self.annotation.load_dataframe()
        for index, row in annotation_df.iterrows():
            with rasterio.open(self.scl_path) as src:
                out_image, out_transform = mask(src, [row["geometry"]], crop=True)
                annotation_scl[row["id"]] = out_image.flatten()
        return annotation_scl

    @bands_required   
    def calculate_ndvi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        red = SatelliteBand(band_name="B04", band_path=self.bands["B04"])
        nir = SatelliteBand(band_name="B08", band_path=self.bands["B08"])
        index = (nir.array.astype(float) - red.array.astype(float)) / (nir.array + red.array)
        index = np.nan_to_num(index)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': red.meta["crs"],
                'transform': red.meta["transform"],
                'compress': 'lzw',
                'nodata': None
            }
            index_path = self.folder / f"S2_{self.date.strftime('%Y%m%d')}_NDVI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.indizes["NDVI"] = index_path
        return index
    
    @band_management_decorator
    def calculate_ndwi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        nir =  SatelliteBand(band_name="B08)", band_path=self.bands["B08"])
        swir1 = SatelliteBand(band_name="B11)", band_path=self.bands["B11"]).resample(10)
        index = (nir.array.astype(float) - swir1.array.astype(float)) / (nir.array + swir1.array)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': nir.meta["crs"],
                'transform': nir.meta["transform"],
                'compress': 'lzw',
                'nodata': None
            }
            index_path = self.folder / f"S2_{self.date.strftime('%Y%m%d')}_NDWI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.indizes["NDWI"] = index_path
        return index
    
    @band_management_decorator
    def open_bands_with_scl(self, band_ids):
        band_files = {}
        for band_id in band_ids: 
            if band_id in self._loaded_bands.keys():
                band_files[band_id] =  rasterio.open(self._loaded_bands[band_id].path)
            if "SCL" in band_ids:
                scl = SatelliteBand(band_name="SCL", band_path=self.scl_path)
                scl.resample(10)
                band_files["SCL"] = rasterio.open(scl.path)
        return band_files

    @band_management_decorator
    def extract_band_data_of_annotations(self, band_ids):
        band_files = self.open_bands_with_scl(band_ids)
        annotation_df = self.annotation.load_and_transform_to_band_crs(next(iter(band_files.values())))
        anns_band_data = {}
        for index, row in annotation_df.iterrows():
            ann_bands_data = extract_band_data_for_annotation(annotation=row, band_files=band_files)
            ann_bands_data["NDVI2"] = extract_nearby_ndvi_data(annotation=row, band_files=band_files)
            anns_band_data[row["id"]] = ann_bands_data
        [band.close() for band in band_files.values()]
        return anns_band_data
    
    @band_management_decorator
    def extract_band_data_of_annotations(self, indizes=False):
        if not self.is_stacked():
            self.stack_bands()

        if indizes:
            ndvi_path = self.folder / f"S2_{self.date.strftime('%Y%m%d')}_NDVI.tif"
            ndwi_path = self.folder / f"S2_{self.date.strftime('%Y%m%d')}_NDWI.tif"

        annotation_df = self.annotation.load_and_transform_to_band_crs(next(iter(band_files.values())))
        anns_band_data = {}
        for index, row in annotation_df.iterrows():
            ann_bands_data = extract_band_data_for_annotation(annotation=row, band_files=band_files)
            ann_bands_data["NDVI2"] = extract_nearby_ndvi_data(annotation=row, band_files=band_files)
            anns_band_data[row["id"]] = ann_bands_data
        [band.close() for band in band_files.values()]
        return anns_band_data
 
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
     
    def __init__(self, folder_of_satellite_image, annotation_shp, date):
        super().__init__()   
        self.folder = Path(folder_of_satellite_image)
        self.band_files_by_id = self._initialize_rasters()   
        self.annotation = SatelliteImageAnnotation(annotation_shp) 
        self.meta = self._load_meta()
        self.date = date

    def _load_meta(self):
        meta_json_path = [entry for entry in self.folder.iterdir() if entry.is_file() and entry.suffix == '.json'][0]
        with open(meta_json_path, "r") as file:
            meta = json.load(file)
        return meta
    
    def _initialize_rasters(self):
        raster_files_dir = {}
        file_paths = [entry for entry in self.folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for raster_filepath in file_paths:
            band_id = self._extract_band_info(str(raster_filepath))
            if band_id:       
                raster_files_dir[band_id.upper()] = raster_filepath
        return raster_files_dir

    def _extract_band_info(self, file_path): 
        pattern = r"(vh|vv)"
        match = re.search(pattern, file_path)
        return match.group(1) if match else None

    @band_management_decorator
    def extract_band_data_of_all_annotations(self, band_ids):
        band_files = {band_id: rasterio.open(self._loaded_bands[band_id].path) for band_id in band_ids if band_id in self._loaded_bands}
        anns_band_data = {}
        annotation_df = self.annotation.load_dataframe()

        for band_id, band in band_files.items():
            if annotation_df.crs != band.crs:
                annotation_df = annotation_df.to_crs(band.crs)

        for index, row in annotation_df.iterrows():
            geometry_bounds = row["geometry"].bounds
            ann_bands_data = {}

            for band_id, band in band_files.items():
                # If the geometry is not within the raster bounds, continue with the next annotation
                if not (band.bounds[0] <= geometry_bounds[2] and band.bounds[2] >= geometry_bounds[0] 
                        and band.bounds[1] <= geometry_bounds[3] and band.bounds[3] >= geometry_bounds[1]):
                    continue
                ann_band_data, _ = mask(band, [row["geometry"]], crop=True)   
                if 'nodata' in band.meta and band.meta['nodata'] is not None:
                    ann_band_masked_data = np.ma.masked_values(ann_band_data, band.meta['nodata'])
                    mean_value = float(ann_band_masked_data.mean()) if ann_band_masked_data.count() > 0 else -9999.0
                else:
                    mean_value = float(ann_band_data.mean()) if ann_band_data.size > 0 else -9999.0
                ann_bands_data[band_id] = mean_value
            
            if ann_bands_data:
                ann_bands_data["pixel_count"] = ann_band_data.size
                ann_bands_data["flight_direction"] = self.meta["properties"]["sat:orbit_state"]
                anns_band_data[row["id"]] = ann_bands_data
        [band.close() for band in band_files.values()]
        return anns_band_data

    @band_management_decorator
    def calculate_cross_ratio(self, save=True):
        vv = self._loaded_bands["VV"]
        vh = self._loaded_bands["VH"]
        index = vh/vv
        if save:
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
            index_path = self.folder / f"S1_{self.date.strftime('%Y%m%d')}_CR.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.add_band("PD", index_path)
        return index

    @band_management_decorator
    def calculate_polarization_difference(self, save=True):
        vv = self._loaded_bands["VV"]
        vh = self._loaded_bands["VH"]
        index = np.nan_to_num(vv-vh)
        if index:
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
            index_path = self.folder / f"S1_{self.date.strftime('%Y%m%d')}_PD.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.add_band("PD", index_path)
        return index
    
    def plot_cross_ratio(self):
        try:
            cr = np.moveaxis(self.load_band("CR"),0,-1)
            plt.imshow(cr)
            plt.title(f"Cross ratio of sentinel-1 image on date: {self.date}")
            plt.show()
            plt.savefig("cr.png")
        except ValueError:
            raise ValueError(f"An error occurred while plotting cross ratio: {ValueError}. Please make sure that specific index is calcluated and added to bands.")
