from .band import SatelliteBand
import rasterio
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
from .utils import buffer_ann_and_extract_scl_data
from rasterio.windows import Window, transform
import numpy as np
from rasterio.mask import mask

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
        self.mask_path = self._find_valid_mask_file()
        self.dataframe = self._load_dataframe()
        self._transform_dataframe_to_image_crs()

    def _find_valid_mask_file(self, resolution=10):
        """
        Find the first valid mask file within the satellite image folder that matches the specified resolution.

        Parameters:
        - resolution (int): The desired resolution of the mask file in meters. Defaults to 10.

        Returns:
        - Path: The path to the first valid mask file found that matches the criteria, or None if no such file exists.
        """
        mask_suffixes = ['.tif', '.tiff']
        mask_keyword = f"{resolution}m_mask" if resolution else "mask"
        files = [file for file in self.image.folder.iterdir() if file.is_file() and file.suffix.lower() in mask_suffixes and mask_keyword in file.stem.lower()]
        return files[0] if files else None

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
        with rasterio.open(self.image.path) as src:
            image_crs = src.crs
        if self.dataframe.crs != image_crs:
            self.dataframe.to_crs(image_crs, inplace=True)

    def rasterize(self, resolution):
        """
        Rasterize the annotations based on a specified resolution, updating or creating a mask file.

        Parameters:
        - resolution (int): The resolution in meters to rasterize the annotations.

        Returns:
        - Path: The path to the rasterized mask file.
        """
        if not self.mask_path:
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
        band_path = self.image.find_band_by_res(resolution)
        if band_path is None:
            raise ValueError(f"Resolution {resolution} not found in image bands.")

        with rasterio.open(band_path) as src:
            raster_meta = src.meta.copy()
            raster_meta.update({'dtype': 'uint8', 'count': 1})

        geometries = self.dataframe["geometry"].values
        mask_filepath = self.image.folder / f"{self.image.name}_{resolution}m_mask.tif"

        with rasterio.open(mask_filepath, 'w', **raster_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)

        self.mask_path = mask_filepath

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

    def create_spectral_signature(self, band_ids, filtering=False):
        """
        Generates spectral signatures for annotations, extracting data from specified
        bands of satellite imagery. Optionally filters annotations based on pixel quality.

        Parameters:
        - band_ids (list of str): Identifiers for the bands from which data will be extracted.
        - filtering (bool): Whether to filter annotations based on the quality of pixels in the "SCL" band.

        Returns:
        - dict: Mapping each annotation ID to its spectral signature across specified bands.
        """
        if filtering and "SCL" not in band_ids:
            band_ids.append("SCL")  # Ensure "SCL" is included for filtering
        
        [idx_function() for idx, idx_function in self.image.index_functions.items() if idx in band_ids]
        opened_bands = {}
        for band_id, band_path in self.image.bands.items():
            if band_id in band_ids:
                band = SatelliteBand(band_name=band_id, band_path=band_path).resample(10)
                opened_bands[band_id] = rasterio.open(band.path)
        
        anns_band_data = {}
        for index, row in self.dataframe.iterrows():
            ann_bands_data = {}
            for band_id, band in opened_bands.items():
                try:
                    if band_id == "SCL":
                        ann_band_data, _ = mask(band, [row["geometry"]], crop=True, nodata=99) # we need to define a new noData value outside 0-12
                        unique, counts = np.unique(ann_band_data.flatten(), return_counts=True)
                        ann_bands_data[band_id] = {int(u): int(c) for u, c in zip(unique, counts)}
                    else:
                        ann_band_data, _ = mask(band, [row["geometry"]], crop=True, nodata=-9999)
                        valid_data = ann_band_data[ann_band_data > 0] # filters out the nodata values 
                        ann_bands_data[band_id] = np.mean(valid_data) if valid_data.size > 0 else 0
                except Exception as e:
                    print(f"Error processing band {band_id}: {e}")
                    ann_bands_data[band_id] = None

            anns_band_data[row["id"]] = ann_bands_data
        
        if filtering:
            anns_band_data = self.filter_ann_by_bad_pixels(anns_band_data)
        [band.close() for band in opened_bands.values()]

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

    def get_ndvi_around_polygon(self, buffer_distance=10):
        """
        Calculate the mean Normalized Difference Vegetation Index (NDVI) around each polygon (annotation)
        in the satellite image's associated dataframe. This method applies a buffer to each polygon
        and then masks the NDVI band with this buffered area, considering only the areas marked as
        vegetation in the SCL (Scene Classification Layer) band. The mean NDVI value is computed for
        areas within the buffer that are classified as vegetation.

        Parameters:
        - buffer_distance (int): The distance (in the same units as the satellite image's projection) to extend
        around each polygon to create a buffer for NDVI calculation. Defaults to 10.

        Returns:
        - dict: A dictionary where each key is an annotation ID from the dataframe, and each value is another
        dictionary with a single key-value pair. This pair's key is 'NDVI_around', and its value is the mean NDVI
        calculated around the buffered polygon. If no vegetation is detected within the buffered area, the NDVI
        value is set to 0.
        """
        # 
        self.image.calculate_ndvi()
        scl_band = SatelliteBand(band_name="SCL", band_path=self.image.bands["SCL"]).resample(10)
        around_anns_ndvi_data = {} 
        for index, row in self.dataframe.iterrows():
            polygon_around_ann, scl_mask_around_ann = buffer_ann_and_extract_scl_data(
                ann_polygon=row["geometry"], 
                scl_band=scl_band, 
                scl_keys=[4], # stands for vegetation
                buffer_distance=buffer_distance)  
            with rasterio.open(self.image.bands["NDVI"],  "r") as src:          
                ndvi, _ = mask(src, [polygon_around_ann], crop=True)
            ndvi_masked = ndvi[scl_mask_around_ann] # only use values where vegetation is in the scl layer
            ndvi_around_ann = np.mean(ndvi_masked) if np.any(scl_mask_around_ann) else 0
            around_anns_ndvi_data[row["id"]] = {"NDVI_around": ndvi_around_ann}
        return around_anns_ndvi_data
    
    #     def create_spectral_signature_in_parallel(self, band_ids, chunks=100):
    #     """
    #     Creates spectral signatures for annotations in parallel, using specified satellite image bands.

    #     This method transforms annotations to match the image CRS, splits the annotations into chunks,
    #     and processes each chunk in parallel to calculate spectral signatures.

    #     Parameters:
    #     - band_ids (list of str): List of band identifiers to include in the spectral signature calculation.
    #     - chunks (int): The number of chunks to split the annotations into for parallel processing.

    #     Returns:
    #     - dict: A dictionary mapping annotation IDs to their calculated spectral signatures.
    #     """
    #     self.transform_annotations_to_image_crs()
    #     dataframe_chunks = split_dataframe(self.dataframe, chunks)
    #     selected_band_paths = {band_id: band_path for band_id, band_path in self.image.bands.items() if band_id in band_ids}
        
    #     progress_bar = tqdm(total=len(self.dataframe), desc="Creating spectral signatures")
        
    #     with ProcessPoolExecutor(max_workers=available_workers()) as executor:
    #         futures = {executor.submit(self.process_chunk, chunk, selected_band_paths): len(chunk) for chunk in dataframe_chunks}
    #         anns_band_data = {}
    #         for future in as_completed(futures):
    #             chunk_results = future.result()
    #             anns_band_data.update(chunk_results)
    #             # Update the progress bar by the number of annotations processed in the chunk
    #             progress_bar.update(futures[future])
                
    #     progress_bar.close()
    #     return anns_band_data

    # @staticmethod
    # def process_chunk(dataframe_chunk, band_paths):
    #     """
    #     Processes a chunk of annotations to calculate spectral signatures based on provided band paths.

    #     This method opens each specified band file, masks the band data with the annotation geometries,
    #     and calculates various statistics (e.g., mean values, unique counts) for the spectral signature.

    #     Parameters:
    #     - dataframe_chunk (pandas.DataFrame): A chunk of the dataframe containing annotations to process.
    #     - band_paths (dict): A dictionary mapping band identifiers to their file paths.

    #     Returns:
    #     - dict: A dictionary mapping each annotation ID in the chunk to its calculated spectral signature.
    #     """
    #     anns_band_data = {}
    #     opened_bands = {band_id: rasterio.open(band_path) for band_id, band_path in band_paths.items()}
        
    #     for index, row in dataframe_chunk.iterrows():
    #         ann_bands_data = {}
    #         for band_id, band in opened_bands.items():
    #             ann_band_data, _ = mask(band, [row["geometry"]], crop=True)
    #             if band_id == "SCL":
    #                 unique, counts = np.unique(ann_band_data.flatten(), return_counts=True)
    #                 ann_bands_data[band_id] = {int(u): int(c) for u, c in zip(unique, counts)}
    #             elif band_id in ["NDVI", "NDWI"]:
    #                 ann_bands_data[band_id] = np.mean(ann_band_data[ann_band_data > 0]) if np.any(ann_band_data > 0) else 0
    #             else:
    #                 ann_bands_data[band_id] = np.mean(ann_band_data)
    #         anns_band_data[row["id"]] = ann_bands_data
    
    #     [band.close() for band in opened_bands.values()]
    #     return anns_band_data
