import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.windows import Window
from .utils import band_management_decorator, pad_patch, get_metadata_of_window, extract_S2band_info, extract_band_number
from .band import SatelliteBand
from .annotation import SatelliteImageAnnotation
from pathlib import Path
import json
from datetime import datetime

class SatelliteImage:
    def __init__(self, folder):
        self.folder = folder # folder of all files correlated to the image like bands, patches etc. 
        self.index_functions = None
        self.bands = {} # dict with "all" bands in self.folder - regular bands + indizes + SCL 
        self.name = None # S2_mspc_l2a_20190509_B02 - function that extract this nam
        self.date = None # date of image
        self.path = None  # Path without indices
        
    def __getitem__(self, band_id):
        try:
            raster_filepath = self.bands[band_id]
            if raster_filepath:
                return SatelliteBand(band_id, raster_filepath)
        except KeyError:
            raise ValueError(f"Band id {band_id} not found.")

    def get_number_of_channels(self):
        if self.path:
            with rasterio.open(self.path, "r") as src:
                return src.count

    def find_band_by_res(self, target_res):
        for band_id in self.bands:
            band = self[band_id]  # Use __getitem__ to get the SatelliteBand instance
            resolution = band.get_resolution() # type: ignore # (10,10)
            if resolution[0] == target_res:
                return band
        return None
    
    def create_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        with rasterio.open(self.path) as src:
            step_size = patch_size - overlay
            for i in range(0, src.width,  step_size):
                for j in range(0, src.height, step_size):
                    args = [j, i, patch_size, patch_size]
                    window = Window(*args)
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
        super().__init__(folder)
        self.folder = folder
        self.bands = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = datetime.strptime(self.name.split("_")[-1], "%Y%m%d").date()
        self.index_functions = {"NDVI": self.calculate_ndvi, "NDWI": self.calculate_ndwi}
        self.path = self.folder / f"{self.name}.tif"
        
    def _initialize_name(self):
        raster_filepath = next(iter(self.bands.values()))
        band_name = raster_filepath.stem
        return "_".join(band_name.split("_")[:-1])

    def _initialize_bands(self):
        raster_files_dir = self._filter_raster_files()
        return self._sort_raster_filepaths(raster_files_dir)

    def _filter_raster_files(self):
        raster_files_dir = {}
        for raster_filepath in self.folder.glob('*.tif'):
            band_id = extract_S2band_info(raster_filepath.name)
            if band_id:
                raster_files_dir[band_id] = raster_filepath
        return raster_files_dir

    def _sort_raster_filepaths(self, raster_files_dir):
        sorted_keys = sorted(raster_files_dir.keys(), key=extract_band_number)
        return {k: raster_files_dir[k] for k in sorted_keys}

    def stack_bands(self, resolution=10, include_indizes=True):
        """
        Stacks all bands and indices (if requested) of the satellite image into a single multi-band raster file.
        Args:
            resolution (int): The target resolution for resampling all bands. Defaults to 10.
            include_indizes (bool): If True, indices like NDVI and NDWI are calculated, added to the bands,
                                    and included in the stacking process. Defaults to False.
        Returns:
            Path: The path to the stacked image file. This will either be the base path or the index path
                of the SatelliteImage object, depending on whether indices were included.      
        Raises:
            FileNotFoundError: If no band matches the desired resolution.
            Exception: For any other unexpected errors during the stacking process.
        """
        # Dynamic path determination based on whether indices are to be included
        if not self.path.exists() or (include_indizes and self.get_number_of_channels() != 13):
            if include_indizes:
                self.calculate_all_indizes()

            band_with_desired_res = self.find_band_by_res(resolution)
            if band_with_desired_res:            
                with rasterio.open(band_with_desired_res.path) as src:
                    meta = src.meta.copy()
                    meta['count'] = len(self.bands)
                    meta['dtype'] = 'uint16'  

            with rasterio.open(self.path, 'w', **meta) as dst:
                band_descriptions = []
                for i, (band_id, band_filepath) in enumerate(self.bands.items(), start=1):
                    band = SatelliteBand(band_id, band_filepath)
                    band_arr = band.resample(resolution).load_array()
                    dst.write(np.squeeze(band_arr), i)
                    band_descriptions.append(band_id)
                dst.descriptions = tuple(band_descriptions)

        return self

    def calculate_bad_pixel_ratio(self):
        scl_path = self.bands["SCL"]
        if scl_path:
            with rasterio.open(scl_path) as src:
                slc = src.read()
            bad_pixel_count = np.isin(slc, [0,1,2,3,8,9,10,11])
            bad_pixel_ratio = np.mean(bad_pixel_count) * 100
            return bad_pixel_ratio

    # def is_quality_acceptable(self, bad_pixel_ratio=15):
    #     if not self.bad_pixel_ratio:
    #         self.calculate_bad_pixel_ratio()
    #     return self.bad_pixel_ratio < bad_pixel_ratio

    def calculate_all_indizes(self):
        raster_files_dir = self._filter_raster_files()
        if not "NDVI" in raster_files_dir:
            self.calculate_ndvi()
        if not "NDWI" in raster_files_dir:
            self.calculate_ndwi()
        return self

    def calculate_ndvi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        red = SatelliteBand(band_name="B04", band_path=self.bands["B04"])
        nir = SatelliteBand(band_name="B08", band_path=self.bands["B08"])
        red_arr, nir_arr = red.load_array().astype(float), nir.load_array().astype(float)
        index = (nir_arr - red_arr) / (nir_arr + red_arr)
        index = np.nan_to_num(index)
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': red.load_meta()["crs"],
                'transform': red.load_meta()["transform"],
                'compress': 'lzw',
                'nodata': np.NaN
            }
            index_path = self.folder / f"{self.name}_NDVI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.bands["NDVI"] = index_path
        return index
    
    def calculate_ndwi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        nir =  SatelliteBand(band_name="B08)", band_path=self.bands["B08"])
        swir1 = SatelliteBand(band_name="B11)", band_path=self.bands["B11"]).resample(10)
        nir_arr, swir1_arr = nir.load_array().astype(float), swir1.load_array().astype(float)
        index = ((nir_arr - swir1_arr) / (nir_arr + swir1_arr))
        if save:
            profile = {
                'driver': 'GTiff',
                'height': index.shape[1],
                'width': index.shape[2],
                'count': 1,
                'dtype': index.dtype,
                'crs': nir.load_meta()["crs"],
                'transform': nir.load_meta()["transform"],
                'compress': 'lzw',
                'nodata': np.NaN
            }
            index_path = self.folder / f"{self.name}_NDWI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.bands["NDWI"] = index_path
        return index
    
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
