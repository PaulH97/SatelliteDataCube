from .band import SatelliteBand
import rasterio
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform
import numpy as np
from rasterio.mask import mask
from shapely import MultiPolygon
from tqdm import tqdm
import random

class SatelliteImageAnnotation:

    def __init__(self, satellite_image, shapefile_path):
        """
        Initialize a SatelliteImageAnnotation object. Each satellite image should have a corresponding annotation. 
        The purpose of thid class is to handle some common preprocessing methods for using the data in an machine learning workflow.

        Parameters:
        - satellite_image: An object representing a satellite image, expected to have attributes like a folder path
        and methods for finding image bands.
        - shapefile_path: A string or Path object pointing to the shapefile location. This shapefile contains
        the polygons (annotations) to be used on the satellite image.

        Attributes initialized include the satellite image object, shapefile path, mask path determined by
        finding a valid mask file, and a GeoDataFrame loaded from the shapefile.
        """
        self.image = satellite_image
        self.shp_path = Path(shapefile_path)
        self.mask_path = Path(self.image.folder, f"{self.image.name}_mask.tif")
        self.gdf = self._load_dataframe()
        self._transform_dataframe_to_image_crs()

    def _find_or_create_mask_file(self, resolution=10):
        """
        Find the first valid mask file within the satellite image folder that matches the specified resolution.

        Parameters:
        - resolution (int): The desired resolution of the mask file in meters. Defaults to 10.

        Returns:
        - Path: The path to the first valid mask file found that matches the criteria, or None if no such file exists.
        """
        mask_suffixes = ['.tif', '.tiff']
        mask_keyword = "mask"
        files = [file for file in self.image.folder.iterdir() if file.is_file() and file.suffix.lower() in mask_suffixes and mask_keyword in file.stem.lower()]        
        return files[0] if files else self.rasterize(resolution)

    def _load_dataframe(self):
        """
        Load the annotations from the shapefile path into a GeoDataFrame and repair any invalid geometries.

        Returns:
        - GeoDataFrame: A GeoDataFrame containing the loaded and potentially repaired geometries from the shapefile.
        """
        df = gpd.read_file(self.shp_path)
        df["geometry"] = df["geometry"].apply(self._repair_geometry)
        return df

    @staticmethod
    def _repair_geometry(geom):
        """
        Repair an invalid geometry using a zero-width buffer operation, or return None if the geometry is missing.

        Parameters:
        - geom: The geometry object to be repaired.

        Returns:
        - Geometry: The repaired geometry, or None if the input geometry was None.
        """
        if geom is None or not geom.is_valid:
            return geom.buffer(0) if geom else None
        return geom

    def _transform_dataframe_to_image_crs(self):
        """
        Transform the annotations in the dataframe to match the coordinate reference system (CRS) of the satellite image.
        """
        image_path = self.image.path if self.image.path.exists() else self.image.index_path
        with rasterio.open(image_path) as src:
            image_crs = src.crs
        if self.gdf.crs != image_crs:
            self.gdf.to_crs(image_crs, inplace=True)

    def select_ann_within_mask(self, extent, overlap_percentage=50):
        ann_temp_gdf = self.gdf.copy()
        ann_temp_gdf['intersection_area'] = ann_temp_gdf.geometry.intersection(extent).area
        ann_temp_gdf['original_area'] = ann_temp_gdf.geometry.area
        ann_temp_gdf['overlap_percentage'] = (ann_temp_gdf['intersection_area'] / ann_temp_gdf['original_area']) * 100
    
        return ann_temp_gdf[ann_temp_gdf['overlap_percentage'] >= overlap_percentage]

    def rasterize(self, resolution):
        """
        Rasterize the annotations based on a specified resolution, updating or creating a mask file.

        Parameters:
        - resolution (int): The resolution in meters to rasterize the annotations.

        Returns:
        - Path: The path to the rasterized mask file.
        """
        if not self.mask_path.exists():
            self._create_mask(resolution)
        return self.mask_path

    def _create_mask(self, resolution):
        """
        Create a mask file from the annotations at the specified resolution.

        Parameters:
        - resolution (int): The resolution in meters for the mask.

        Raises:
        - ValueError: If no band is found matching the specified resolution in the image bands.
        """
        self._transform_dataframe_to_image_crs()
        band = self.image.find_band_by_res(resolution)

        if band is None:
            raise ValueError(f"Resolution {resolution} not found in image bands.")

        with rasterio.open(band.path) as src:
            raster_meta = src.meta.copy()
            raster_meta.update({'dtype': 'uint8', 'count': 1})

        geometries = self.gdf["geometry"].values

        with rasterio.open(self.mask_path, 'w', **raster_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)

    def create_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        """
        Create patches from the mask file, potentially with overlay and padding, and save them to a directory.

        Parameters:
        - patch_size (int): The size of the square patches.
        - overlay (int): The overlay size between adjacent patches. Defaults to 0.
        - padding (bool): Whether to pad patches that are smaller than the patch size. Defaults to True.
        - output_dir (Path or str, optional): The directory to save the patches. Defaults to a subdirectory
        "patches/MSK" within the image folder.

        Returns:
        - Path: The directory where the patches were saved.
        """
        output_dir = Path(output_dir) if output_dir else self.image.folder / "patches" / "MSK"
        output_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.mask_path) as src:
            patches_folder = self._generate_patches(src, patch_size, overlay, padding, output_dir)
        return patches_folder

    def _generate_patches(self, src, patch_size, overlay, padding, output_dir):
        """
        Helper function to generate and save patches from the source raster based on the specified parameters.

        Parameters:
        - src: The source rasterio object to generate patches from.
        - patch_size (int): The size of the square patches.
        - overlay (int): The overlay size between adjacent patches.
        - padding (bool): Whether to pad patches that are smaller than the patch size.
        - output_dir (Path): The directory to save the patches.

        Returns:
        - Path: The directory where the patches were saved.
        """
        step_size = patch_size - overlay
        for i in range(0, src.width, step_size):
            for j in range(0, src.height, step_size):
                patch, window = self._extract_patch(src, i, j, patch_size, padding)
                if patch is not None:
                    self._save_patch(patch, window, src, i, j, output_dir)
        return output_dir

    @staticmethod
    def _extract_patch(src, i, j, patch_size, padding):
        """
        Extract a patch from the source at the specified location and size, optionally applying padding.

        Parameters:
        - src: The source rasterio object.
        - i (int): The horizontal offset for the patch.
        - j (int): The vertical offset for the patch.
        - patch_size (int): The size of the square patch.
        - padding (bool): Whether to apply padding to patches smaller than the specified size.

        Returns:
        - tuple: A tuple containing the extracted patch and its window. If padding is False and the patch is smaller
        than the specified size, returns (None, None).
        """
        args = [i, j, patch_size, patch_size]
        window = Window(*args)
        patch = src.read(window=window)
        if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
            patch = np.pad(patch, ((0, 0), (0, patch_size - patch.shape[1]), (0, patch_size - patch.shape[2])), mode='constant')
        elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
            return None, None
        return patch, window
    
    def _save_patch(self, patch, window, src, i, j, output_dir):
        """
        Save a patch to a file within the specified output directory.

        Parameters:
        - patch: The patch to be saved.
        - window: The window object representing the patch's location and size in the source raster.
        - src: The source rasterio object.
        - i (int): The horizontal index of the patch.
        - j (int): The vertical index of the patch.
        - output_dir (Path): The directory where the patch will be saved.

        The saved file will be named following the pattern "patch_{i}_{j}.tif".
        """
        patch_meta = src.meta.copy()
        patch_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': transform(window, src.transform)
        })
        patch_filename = f"patch_{i}_{j}.tif"
        patch_filepath = output_dir / patch_filename
        
        with rasterio.open(patch_filepath, 'w', **patch_meta) as patch_dst:
            patch_dst.write(patch)

    def create_spectral_signature(self, band_ids):
        """
        Generates spectral signatures for annotations, extracting data from specified
        bands of satellite imagery. Optionally filters annotations based on pixel quality.

        Parameters:
        - band_ids (list of str): Identifiers for the bands from which data will be extracted.
        - filtering (bool): Whether to filter annotations based on the quality of pixels in the "SCL" band.

        Returns:
        - dict: Mapping each annotation ID to its spectral signature across specified bands.
        """
        opened_bands = {}
        for band_id, band_path in self.image.bands.items():
            if band_id in band_ids:
                band = SatelliteBand(band_name=band_id, band_path=band_path).resample(10)
                opened_bands[band_id] = rasterio.open(band.path)
        
        anns_band_data = {}
        for index, row in self.gdf.iterrows():
            ann_bands_data = {}
            for band_id, band in opened_bands.items():
                try:
                    ann_band_data, _ = mask(band, [row["geometry"]], crop=True, nodata=-9999)
                    valid_data = ann_band_data[ann_band_data > 0] # filters out the nodata values 
                    ann_bands_data[band_id] = np.mean(valid_data) if valid_data.size > 0 else 0
                except Exception as e:
                    print(f"Error processing band {band_id}: {e}")
                    ann_bands_data[band_id] = None

            anns_band_data[row["id"]] = ann_bands_data

        return anns_band_data
    
    @staticmethod
    def filter_ann_by_bad_pixels(anns_band_data, bad_pixel_ratio=30):
        """
        Filters annotations by their bad pixel ratio in the "SCL" band. If the ratio of bad pixels
        to total pixels exceeds the given threshold, the annotation's data is replaced with an empty
        dictionary, effectively retaining the annotation ID but indicating its data is unreliable.

        Parameters:
        - anns_band_data (dict): The dictionary containing annotation IDs as keys and corresponding
        band data as values.
        - bad_pixel_ratio (float, optional): The threshold ratio (percentage) of bad pixels above which
        the annotation data is considered unreliable. Defaults to 50.

        Returns:
        - dict: The filtered annotations dictionary with annotations having a high bad pixel ratio
        having their data replaced with an empty dictionary.
        """
        filtered_anns_band_data = {}
        for ann_id, bands in anns_band_data.items():
            ls_pixel_count = sum([count for count in bands["SCL"].values()])
            bad_pixel_count = sum([count for scl_key, count in bands["SCL"].items() if int(scl_key) in [0,1,2,3,8,9,10,11]])
            ann_bad_pixel_ratio = (bad_pixel_count/ls_pixel_count) * 100
            if ann_bad_pixel_ratio <= bad_pixel_ratio:
                bands.pop("SCL", None)
                filtered_anns_band_data[ann_id] = bands 
            else:  
                filtered_anns_band_data[ann_id] = {}
        return filtered_anns_band_data

    def preprocess_annotation_zones(self, buffer_distance=20):
        annotation_zones = MultiPolygon([geom.buffer(buffer_distance) for geom in self.gdf.geometry])
        annotation_zones = MultiPolygon([geom.buffer(0) for geom in self.gdf.geometry if not geom.is_valid])
        return annotation_zones.simplify(1.0, preserve_topology=True)

    @staticmethod
    def calculate_annotation_buffer_ndvi(geometry, annotation_zones, scl_src, ndvi_src, ann_pixel_count):
        buffer_dist = 100  # Starting buffer distance
        num_of_pixels = ann_pixel_count * 5
        while True:
            ann_large_buffer = geometry.buffer(buffer_dist)
            buffered_zone = ann_large_buffer.difference(annotation_zones)
            scl_data, _ = mask(scl_src, [buffered_zone], crop=True)
            ndvi_data, _ = mask(ndvi_src, [buffered_zone], crop=True)
            vegetation_mask = scl_data[0] == 4  # Targets only vegetation pixels
            ndvi_masked = ndvi_data[0][vegetation_mask].flatten()

            if ndvi_masked.size > 0 and ndvi_masked.size >= num_of_pixels:
                # ndvi_masked = np.random.choice(ndvi_masked, num_of_pixels, replace=False)
                ndvi_mean = np.mean(ndvi_masked)
                ndvi_median = np.median(ndvi_masked)
                break
            elif ndvi_masked.size > 0 and ndvi_masked.size < num_of_pixels:
                # If there's some data but not enough, adjust the buffer and try again
                buffer_dist *= 1.5
            else:
                buffer_dist *= 1.5 
                if buffer_dist > 500:  # Prevent infinite loop
                    ndvi_mean = None 
                    ndvi_median = None
                    break

        return ndvi_mean, ndvi_median
        
    @staticmethod 
    def calculate_annotation_ndvi(geometry, scl_src, ndvi_src, good_pixel_threshold):
        bad_pixel_values = [0, 1, 2, 3, 8, 9, 10, 11]
        scl_ann, _ = mask(scl_src, [geometry], crop=True, nodata=99, all_touched=True)
        ndvi_ann, _ = mask(ndvi_src, [geometry], crop=True)
        
        valid_scl_mask = scl_ann != 99
        valid_ndvi_mask = ~np.isnan(ndvi_ann)
        combined_valid_mask = valid_scl_mask & valid_ndvi_mask
        
        good_pixel_mask = (~np.isin(scl_ann[combined_valid_mask], bad_pixel_values))
        valid_ndvi_ann = ndvi_ann[combined_valid_mask].flatten()

        if good_pixel_mask.size != valid_ndvi_ann.size:
            raise ValueError("Mismatch in good pixel mask and valid NDVI data sizes.")
            
        good_pixel_percentage = (np.sum(good_pixel_mask) / good_pixel_mask.size) * 100.0

        if good_pixel_percentage >= good_pixel_threshold:
            ndvi_masked_ann = valid_ndvi_ann[good_pixel_mask]

            if ndvi_masked_ann.size > 0:
                ndvi_mean_ann = np.nanmean(ndvi_masked_ann)
                ndvi_median_ann = np.nanmedian(ndvi_masked_ann)
            else:
                ndvi_mean_ann = ndvi_median_ann = None
        else:
            ndvi_mean_ann = ndvi_median_ann = None

        ann_pixel_count = np.sum(combined_valid_mask)

        return ndvi_mean_ann, ndvi_median_ann, ann_pixel_count

    def calculate_ndvis_for_dating(self, good_pixel_threshold=70):
        self.image.calculate_ndvi()
        ndvi_path = self.image.bands["NDVI"]
        scl_path = SatelliteBand("SCL", self.image.bands["SCL"]).resample(10).path
        annotation_zones = self.preprocess_annotation_zones()

        anns_ndvi_data = {}
        
        with rasterio.open(scl_path) as scl_src, rasterio.open(ndvi_path) as ndvi_src:
            for _, row in self.gdf.iterrows():

                annotation_ndvi_mean, annotation_ndvi_median, ann_pixel_count = self.calculate_annotation_ndvi(
                    row["geometry"], scl_src, ndvi_src, good_pixel_threshold)

                buffer_ndvi_mean, buffer_ndvi_median = self.calculate_annotation_buffer_ndvi(
                    row["geometry"], annotation_zones, scl_src, ndvi_src, ann_pixel_count)
                                
                anns_ndvi_data[row["id"]] = {
                    "Annotation_NDVI_Mean": annotation_ndvi_mean,
                    "Annotation_NDVI_Median": annotation_ndvi_median,
                    "Buffer_NDVI_Mean": buffer_ndvi_mean,
                    "Buffer_NDVI_Median": buffer_ndvi_median
                }

        return anns_ndvi_data
    
class SatCubeAnnotation:
    def __init__(self, inventory_dir):
        self.inventory_dir = Path(inventory_dir)
        self.folder = self.inventory_dir / "annotations"
        self.name = f"{inventory_dir.name}" 
        self.shp_file = self.folder / f"{self.name}.shp"
        self.mask_path = self.folder / f"{self.name}_mask.tif"
        self.gdf = self._initalize_dataframe()
        
    def _initalize_dataframe(self):
        
        def repair_geometry(geom):
            if geom is None or not geom.is_valid:
                return geom.buffer(0) if geom else None
            return geom

        if self.shp_file.exists():
            gdf = gpd.read_file(self.shp_file)
            gdf["geometry"] = gdf["geometry"].apply(repair_geometry)
            return gdf
        else:
            print("No shapefile found in the annotations folder.")
            return None
    
    def rasterize_annotations(self, raster_meta):
        raster_crs = raster_meta["crs"]
        if self.gdf.crs != raster_crs:
            self.gdf.to_crs(raster_crs, inplace=True)

        if not self.mask_path.exists():
            geometries = self.gdf["geometry"].values
            raster_meta.update({'dtype': 'uint8', 'count': 1})

            with rasterio.open(self.mask_path, 'w', **raster_meta) as dst:
                mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
                dst.write(mask.astype(rasterio.uint8), 1)
        else:
            print("Mask file already exists in the annotations folder.")

        return self.mask_path
