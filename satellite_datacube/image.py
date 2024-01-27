import numpy as np
import rasterio
from matplotlib import pyplot as plt
import re
from rasterio.mask import mask
from .utils import band_management_decorator, find_buffer_with_specific_scl_data
from .band import SatelliteBand
from .annotation import SatelliteImageAnnotation
from pathlib import Path
import json

class SatelliteImage:
    def __init__(self):
        self.band_files_by_id = {}
        self._loaded_bands = {}
        self.annotation = None
        self.array = None
        self.date = None

    def __getitem__(self, band_id):
        if band_id not in self._loaded_bands:
            band_path = self.band_files_by_id.get(band_id)
            if band_path:
                self._loaded_bands[band_id] = SatelliteBand(band_id, band_path)
            else:
                raise ValueError(f"Band path for '{band_id}' not found.")
        return self._loaded_bands[band_id]

    def get_band(self, band_id):
        return self.band_files_by_id[band_id]
        
    def add_band(self, band_id, file_path):
        """
        Add a new SatelliteBand object to the bands dictionary.
        """
        self.band_files_by_id[band_id] = file_path
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
            band_path = self.band_files_by_id.get(band_id)
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
        for band_id in self.band_files_by_id.keys():
            self.load_band(band_id)
        return self._loaded_bands
    
    def unload_all_bands(self):
        self._loaded_bands = {}
    
    @band_management_decorator
    def resample(self, resolution):
        for band_id, band in self._loaded_bands.items():
            band = band.resample(resolution)
            self.band_files_by_id[band_id] = band.path
        return self
    
    @band_management_decorator
    def stack_bands(self):
        try:
            bands_stacked = np.vstack([band.array for band in self._loaded_bands.values()])
            self.array = bands_stacked
        except ValueError:
            raise ValueError(f"An error occurred while stacking bands: {ValueError}. Please make sure that all loaded bands are resampled to the same resolution.")
        return self
    
class Sentinel2(SatelliteImage):
    def __init__(self, folder_of_satellite_bands, annotation_shp, date):
        super().__init__()   
        self.band_folder = Path(folder_of_satellite_bands)
        self.band_files_by_id = self._initialize_bands_path()    
        self.scl_path = self._initialize_scl()
        self.annotation = SatelliteImageAnnotation(annotation_shp)
        self.bad_pixel_ratio = None
        self.date = date
    
    def _initialize_bands_path(self):
        bands_path = {}
        file_paths = [entry for entry in self.band_folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(band_path)
            if band_id:       
                bands_path[band_id] = band_path
        return self._sort_band_paths(bands_path)
    
    def _initialize_scl(self):
        return list(self.band_folder.glob('*SCL.tif'))[0]

    def _sort_band_paths(self, bands_path):
        sorted_keys = sorted(bands_path.keys(), key=self._extract_band_number) # 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12', 'NDVI', 'NDWI', 'SCL
        sorted_bands_paths = {k: bands_path[k] for k in sorted_keys}
        return sorted_bands_paths
    
    @staticmethod
    def _extract_band_info(file_path):
        file_name = file_path.name  
        pattern = r"(B\d+[A-Z]?)\.tif"
        match = re.search(pattern, file_name)
        return match.group(1) if match else None
    
    @staticmethod        
    def _extract_band_number(key):
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

    @band_management_decorator     
    def calculate_ndvi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        red = self._loaded_bands["B04"]
        nir = self._loaded_bands["B08"]
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
            index_path = self.band_folder / f"S2_{self.date.strftime('%Y%m%d')}_NDVI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.add_band("NDVI", index_path)
        return index
    
    @band_management_decorator
    def calculate_ndwi(self, save=True):
        np.seterr(divide='ignore', invalid='ignore')
        nir = self._loaded_bands["B08"]
        swir1 = self._loaded_bands["B11"].resample(10)
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
            index_path = self.band_folder / f"S2_{self.date.strftime('%Y%m%d')}_NDWI.tif"
            with rasterio.open(index_path, 'w', **profile) as dst:
                dst.write(index)
            self.add_band("NDWI", index_path)
        return index
    
    @band_management_decorator
    def extract_band_data_of_all_annotations(self, band_ids):
        band_files = {band_id: rasterio.open(self._loaded_bands[band_id].path) for band_id in band_ids if band_id in self._loaded_bands}
        if "SCL" in band_ids:
            scl = SatelliteBand(band_name="SCL", band_path=self.scl_path)
            scl.resample(10)
            band_files["SCL"] = rasterio.open(scl.path)
        anns_band_data = {}
        annotation_df = self.annotation.load_dataframe()
        for index, row in annotation_df.iterrows():
            ann_bands_data = {}
            polygon_around_annotation, around_ann_scl_data = find_buffer_with_specific_scl_data(row["geometry"], band_files["SCL"], scl_keys=[4] )            
            for band_id, band in band_files.items():
                ann_band_data, _ = mask(band, [row["geometry"]], crop=True)
                around_ann_band_data, _ = mask(band, [polygon_around_annotation], crop=True)
                if band_id == "SCL":
                    unique, counts = np.unique(ann_band_data.flatten(), return_counts=True)
                    ann_bands_data[band_id] = {int(u): int(c) for u, c in zip(unique, counts)}
                elif band_id == "NDVI":   
                    around_ann_band_data = around_ann_band_data[around_ann_scl_data] # only use values where vegetation is in the scl layer
                    ann_band_data[ann_band_data>0]
                    ann_bands_data["NDVI"] = np.mean(ann_band_data[ann_band_data>0]) if np.any(ann_band_data>0) else 0
                    if not np.any(around_ann_scl_data):
                        ann_bands_data["NDVI2"] = 0
                    else:
                        ann_bands_data["NDVI2"] = np.mean(around_ann_band_data) # NDVI values of buffer of annotation 
                else:
                    ann_bands_data[band_id] = ann_band_data.mean()
            anns_band_data[row["id"]] = ann_bands_data
        [band.close() for band in band_files.values()]
        return anns_band_data
 
    @band_management_decorator
    def extract_bands_data_of_all_annotations(self):
        band_ids = [band_id for band_id in self._loaded_bands.keys()] + ["SCL"]
        band_files = {band_id: rasterio.open(band.path) for band_id, band in self._loaded_bands.items()}
        band_files["SCL"] = rasterio.open(self.scl_path)
        
        anns_spectral_signature = {}
        annotation_df = self.annotation.load_dataframe()
        for index, row in annotation_df.iterrows():
            band_data = {}
            for band_id in band_ids:
                src = band_files[band_id]
                out_image, out_transform = mask(src, [row["geometry"]], crop=True)
                if band_id == "SCL":
                    unique, counts = np.unique(out_image.flatten(), return_counts=True)
                    band_data[band_id] = {int(u): int(c) for u, c in zip(unique, counts)}
                else:
                    band_data[band_id] = out_image.mean()
            anns_spectral_signature[row["id"]] = band_data
        for src in band_files.values():
            src.close()
        return anns_spectral_signature

    @band_management_decorator
    def extract_band_data_of_single_annotations(self, annotation_id):
        band_ids = [band_id for band_id in self._loaded_bands.keys()] + ["SCL"]
        bands = [band for band in self._loaded_bands.values()] + [self.scl_path]
        anns_df = self.annotation.load_dataframe()
        ann_spectral_signature = {}
        for index, row in anns_df.iterrows():
            if row["id"] == annotation_id:
                for band_id, band in zip(band_ids, bands):
                    if band_id == "SCL":
                        with rasterio.open(band) as src:
                            out_image, out_transform = mask(src, [row["geometry"]], crop=True)
                            ann_spectral_signature[band_id] = out_image.flatten()
                    else:
                        with rasterio.open(band.path) as src:
                            out_image, out_transform = mask(src, [row["geometry"]], crop=True)
                            ann_spectral_signature[band_id] = out_image.mean()
        return ann_spectral_signature

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
        self.band_folder = Path(folder_of_satellite_image)
        self.band_files_by_id = self._initialize_bands_path()   
        self.annotation = SatelliteImageAnnotation(annotation_shp) 
        self.meta = self._load_meta()
        self.date = date

    def _load_meta(self):
        meta_json_path = [entry for entry in self.band_folder.iterdir() if entry.is_file() and entry.suffix == '.json'][0]
        with open(meta_json_path, "r") as file:
            meta = json.load(file)
        return meta
    
    def _initialize_bands_path(self):
        bands_path = {}
        file_paths = [entry for entry in self.band_folder.iterdir() if entry.is_file() and entry.suffix == '.tif']
        for band_path in file_paths:
            band_id = self._extract_band_info(str(band_path))
            if band_id:       
                bands_path[band_id.upper()] = band_path
        return bands_path

    def _extract_band_info(self, file_path): 
        pattern = r"(vh|vv)"
        match = re.search(pattern, file_path)
        return match.group(1) if match else None

    @band_management_decorator
    def extract_band_data_of_all_annotations(self, band_ids):
        band_files = {band_id: rasterio.open(self._loaded_bands[band_id].path) for band_id in band_ids if band_id in self._loaded_bands}
        anns_band_data = {}
        annotation_df = self.annotation.load_dataframe()
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
            index_path = self.band_folder / f"S1_{self.date.strftime('%Y%m%d')}_CR.tif"
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
            index_path = self.band_folder / f"S1_{self.date.strftime('%Y%m%d')}_PD.tif"
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
