import os
from matplotlib import pyplot as plt
from .image import Sentinel1, Sentinel2
from .annotation import SatelliteImageAnnotation, SatCubeAnnotation
from .utils import get_patch_extent_as_polygon, find_closest_image, save_spectral_signature, load_spectral_signature, pad_patch, log_progress, create_xarray_dataset, available_workers
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from rasterio.windows import Window
import shutil
import rasterio
import pandas as pd
import logging
import traceback
from tqdm import tqdm
import numpy as np
import random
from rasterio.features import shapes
from shapely.geometry import shape as Shape
import geopandas as gpd
import json

class SatelliteDataCube:
    def __init__(self):
        """
        Initializes a new instance of the SatelliteDataCube class.
        
        Attributes:
        - base_folder (str): The base folder where the satellite data is stored.
        - satellite (str): The name of the satellite.
        - satellite_images_folder (str): The specific folder for satellite images.
        - ann_file (str): The annotation file path.
        - images_by_date (dict): A dictionary mapping dates to satellite images.
        """
        self.base_folder = None
        self.satellite = None
        self.satellite_images_folder = None
        self.ann_file = None
        self.images = []
        # self.masks = [] # list of SatelliteImageAnnotation
        self.specSig = None
    
    def _print_initialization_info(self):
        """
        Display detailed initialization information about the data-cube.

        This utility method prints out a summary of the initialization parameters for the data-cube.
        It shows information about the base folder, desired timeseries length, bad pixel limit 
        for satellite images in the timeseries, and the patch size.

        The printout is structured for clarity, with dividers separating different sections 
        and clear labeling for each parameter.

        Returns:
        None

        Example Output:
        ------------------- base_folder_name -------------------
        Initializing data-cube with following parameter:
        - base folder: .../SatelliteDataCube/Chimanimani
        - Start-End: 2018-09-06 00:00:00 -> 2019-09-16 00:00:00
        - Length of data-cube: 90
        """
        divider = "-" * 20
        dates = [image.date for image in self.images]
        dates.sort()
        #print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print(f"{2*divider}")
        print("Initialized data-cube with following parameter:")
        print(f"- Base folder: {self.base_folder}")
        print(f"- Satellite mission: {self.satellite}")
        print(f"- Start-End: {min(dates)} -> {max(dates)}")
        print(f"- Length of data-cube: {len(self.images)}")
        print(f"{2*divider}")
        del dates

    def _pad_images(self, selected_images, total_images_needed):
        additional_needed = total_images_needed - len(selected_images)
        if additional_needed > 0:
            adding_images_front = additional_needed // 2
            adding_images_back = additional_needed - adding_images_front
            images_front = [self.images[0]] * adding_images_front
            images_back = [self.images[-1]] * adding_images_back
            selected_images = images_front + selected_images + images_back
        return selected_images

    def filter_images_by_date(self, pre_post_date, total_images):
        pre_date = datetime.strptime(pre_post_date[0], "%Y-%m-%d").date()
        post_date = datetime.strptime(pre_post_date[1], "%Y-%m-%d").date()
    
        # Use the find_closest_image function to find the closest images to pre and post dates
        pre_event_image = find_closest_image(self.images, pre_date)
        post_event_image = find_closest_image(self.images, post_date)
        
        selected_images = [pre_event_image, post_event_image]
        remaining_slots = total_images - len(selected_images)
        additional_candidates = [img for img in self.images if img.date != pre_date and img.date != post_date]

        if remaining_slots > 0:
            if len(additional_candidates) >= remaining_slots:
                selected_images += random.sample(additional_candidates, remaining_slots)
            else:
                # Use all additional candidates and then pad the selection if necessary
                selected_images += additional_candidates
                selected_images = self._pad_images(selected_images, total_images)
        
        return sorted(selected_images, key=lambda image: image.date)

    def filter_images_by_number(self, total_images):
        unique_months = {(image.date.year, image.date.month) for image in self.images}
        months_count = len(unique_months)
        base_number_per_month = max(int(total_images / months_count), 1)  # Ensure at least one image per month

        # with ProcessPoolExecutor(max_workers=available_workers()) as executor:
        #     ratios = [executor.submit(image.calculate_bad_pixel_ratio()) for image in self.images]
        ratios = [image.calculate_bad_pixel_ratio() for image in self.images]   
        images_ratios = {image: result for image, result in zip(self.images, ratios)}
        all_images_sorted = sorted(self.images, key=lambda image: images_ratios[image])

        selected_images_per_month = {}
        for image in all_images_sorted:
            year_month = (image.date.year, image.date.month)
            if len(selected_images_per_month.get(year_month, [])) < base_number_per_month:
                selected_images_per_month.setdefault(year_month, []).append(image)

        selected_images = [image for images in selected_images_per_month.values() for image in images]

        additional_images = [image for image in all_images_sorted if image not in selected_images]
        additional_needed = min(total_images - len(selected_images), len(additional_images))
        selected_images += random.sample(additional_images, additional_needed)

        # Add images at the beginning/end if the total is still not met
        timeseries_length = len(selected_images)
        if timeseries_length < total_images:
            missing_images = total_images - timeseries_length
            adding_images_front = missing_images // 2
            adding_images_back = missing_images - adding_images_front
            images_front = [self.images[0]] * adding_images_front
            images_back = [self.images[-1]] * adding_images_back
            selected_images = images_front + selected_images + images_back

        return sorted(selected_images, key=lambda image:image.date)
    
    def get_image_dates_range(self):
        dates = [image.date for image in self.images]
        start_date = min(dates)
        end_date = max(dates)
        return start_date, end_date
        
    def create_spectral_signature(self, band_ids, output_dir=None):
        """
        Creates spectral signatures for satellite images processed in parallel.
        
        This function processes each satellite image to extract spectral signatures
        for specified band IDs. The results are saved in a JSON file named after
        the base folder with a "_S2_specSig.json" suffix. If the spectral signature
        file already exists, it is loaded and returned without reprocessing.

        Parameters:
        - band_ids: A list of band IDs for which spectral signatures are to be created.
        - output_dir: Optional. The directory where the spectral signature JSON file
        will be saved. If not specified, a default directory within `self.base_folder`
        named "other" is used.
        - indizes: Optional. Flag indicating whether index-based processing is enabled.

        Returns:
        - A dictionary containing spectral signatures keyed by annotation ID and date.
        """
        output_dir = self.base_folder / "other" if not output_dir else Path(output_dir)
        output_path = output_dir / f"{self.base_folder.name}_specSig.json" 
        if not output_path.exists():
            print(f"Start extracting values for building a spectral signature")
            tasks = [(image, self.ann_file, band_ids) for image in self.images]

            with ProcessPoolExecutor(max_workers=available_workers(reduce_by=5)) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._create_specSig_of_ann_for_image, task): task for task in tasks}
                dc_specSig = {}
                for future in tqdm(as_completed(future_tasks), total=len(future_tasks), desc="Creating spectral signature"):
                    try:
                        img_date, img_specSig, logs = future.result()
                        for ann_id, bands in img_specSig.items():
                            if ann_id not in dc_specSig:
                                dc_specSig[ann_id] = {}
                            dc_specSig[ann_id][img_date] = bands
                        if logs.get("status") != "success":
                            print(f"Task raised following error: {logs.get('error', 'Unknown error')}")
                    except Exception as exc:
                        print(f"Unexpected exception {exc}: {traceback.format_exc()}")
                save_spectral_signature(dc_specSig, output_path)
                return dc_specSig
        else:
            return load_spectral_signature(output_path)

    @staticmethod
    def _create_specSig_of_ann_for_image(task):
        """
        Static method to create spectral signatures for a single satellite image's annotations.
        
        This method is designed to be called in parallel for multiple images. It checks if the spectral
        signature file already exists to avoid duplicate processing and calculates the spectral signatures
        based on specified band IDs.

        Parameters:
        - task (tuple): A tuple containing information required for processing a single image, including
        the image object, annotation file path, band IDs, indices calculation flag, and output file path.

        Returns:
        - tuple: A tuple containing the image date, the calculated spectral signatures for the image,
        and a result dictionary with the status, details, and any errors encountered.
        
        The spectral signatures include various bands and, if specified, additional indices like NDVI.
        Errors and processing details are captured in a result dictionary for logging and debugging.
        """
        image, ann_file, band_ids = task
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        try:
            img_date = image.date.strftime("%Y%m%d")
            ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            img_specSig = ann.create_spectral_signature(band_ids) 
            result["status"] = "success"
            result["details"] = f"Successfully created spectral signature for {image} on {img_date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc() 
        return img_date, img_specSig, result
            
    def rasterize_annotations(self, resolution):
        tasks = [(image, self.ann_file, resolution) for image in self.images]

        with ProcessPoolExecutor(max_workers=available_workers()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._rasterize_annotation, task): task for task in tasks}
            log_progress(future_tasks=future_tasks, desc="Rasterize annotation of images")
        return 
    
    @staticmethod
    def _rasterize_annotation(task):
        image, ann_file, resolution = task
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        try:
            annotation = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            annotation.rasterize(resolution=resolution)
            result["status"] = "success"
            result["details"] = f"Successfully processed IMG patches for {image} on {image.date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()  # Capture full traceback
        return result

    def stack_bands(self, resolution, include_indizes):
        tasks = [(image, resolution, include_indizes) for image in self.images] # necessary to put it in tuple - for serialization 
        with ProcessPoolExecutor(max_workers=available_workers()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._stacking_image_bands, task): task for task in tasks}
            log_progress(future_tasks=future_tasks, desc="Stacking bands of images")
        return 
    
    @staticmethod
    def _stacking_image_bands(task):
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        image, resolution, include_indizes = task
        try:
            image.stack_bands(resolution, include_indizes) 
            result["status"] = "success"
            result["details"] = f"Successfully processed IMG patches for {image} on {image.date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()  # Capture full traceback
        return result

    def _prepare_tasks_for_patches(self, patch_size, overlay, total_images, ann_ratio=0.8):
        print("Preparing single tasks for creating patches, e.g. filter images and balance data")
        ann = SatelliteImageAnnotation(self.images[0], self.ann_file)
        img_tasks, msk_tasks = [], []
        patches_with_ann_count = 0
        patches_without_ann_count = 0
        with rasterio.open(self.images[0].path) as src:
            step_size = patch_size - overlay
            for i in tqdm(range(0, src.width, step_size), desc="Filter timeseries before patch generation"):
                for j in range(0, src.height, step_size):
                    args = [i, j, patch_size, patch_size]
                    window = Window(*args)
                    patch_extent = get_patch_extent_as_polygon(src, window)
                    ann_within_patch = ann.select_ann_within_mask(patch_extent, overlap_percentage=10)
                    
                    total_patches_count = patches_with_ann_count + patches_without_ann_count
                    current_dist = patches_with_ann_count / total_patches_count if total_patches_count > 0 else 1

                    if (not ann_within_patch.empty or (ann_within_patch.empty and current_dist > ann_ratio)):
                        if not ann_within_patch.empty:
                            for idx, row in ann_within_patch.iterrows():
                                patches_with_ann_count += 1 
                                pre_post_date = (row["pre_date"], row["post_date"])
                                selected_images = self.filter_images_by_date(pre_post_date, total_images) 
                                selected_images_paths = [image.path for image in selected_images if image.path.exists()]
                                img_tasks.append((i, j, patch_size, selected_images_paths))
                                msk_tasks.append((i, j, patch_size, [ann.mask_path])) # 
                        else:
                            patches_without_ann_count += 1
                            selected_images = random.sample(self.images, total_images)
                            selected_images_paths = [image.path for image in selected_images if image.path.exists()]
                            img_tasks.append((i, j, patch_size, selected_images_paths))
                            msk_tasks.append((i, j, patch_size, [ann.mask_path]))    
        
        return img_tasks, msk_tasks

    def create_patches(self, patch_size, total_images, overlay=0, padding=True, output_dir=None, ann_ratio=0.8):
        """
        Splits satellite imagery and masks into smaller, overlapped patches for processing or analysis, and saves them to a specified directory.

        This method divides a set of satellite images with there coressponding single global mask into patches of a specified size, optionally with an overlay between adjacent patches and padding at the image edges. It is designed to handle large satellite data cubes by processing images in parallel, improving efficiency for large-scale analysis.

        Parameters:
        - patch_size (int): The size of each square patch (in pixels). Each patch will be patch_size x patch_size.
        - total_images(int): The number of images (timesteps in SITS) should be included
        - overlay (int, optional): The overlap size (in pixels) between adjacent patches. Defaults to 0, meaning no overlap.
        - padding (bool, optional): If True, adds padding to the patches at the edges of the images. Defaults to True.
        - output_dir (Path or str, optional): The directory where the patches will be saved. If not specified, patches will be saved in a default directory structure: base_folder/patches/satellite.

        The method first gathers paths of satellite images to process, calculates the number of patches based on the specified patch size and overlay, and then processes each patch in parallel, utilizing all available CPU cores. Each patch is extracted from the satellite images, optionally padded, and saved to the output directory in a 4D image format suitable for further analysis.

        Returns:
        None. The patches are saved directly to the specified output directory.

        Examples:
        # Basic usage with default settings
        SatelliteDataCube.create_img_patches(patch_size=256)

        # Custom usage with an overlay of 10 pixels, padding disabled, and specifying an output directory
        SatelliteDataCube.create_img_patches(patch_size=512, overlay=10, padding=False, output_dir="path/to/output/dir")
        """
        output_dir = output_dir if output_dir else Path(self.base_folder, "patches")

        if not output_dir.exists():    
            img_tasks, msk_tasks = self._prepare_tasks_for_patches(patch_size, overlay, total_images, ann_ratio=ann_ratio) 
            img_tasks_extended, msk_tasks_extended = [], []
            for idx, (img_task, msk_task) in enumerate(zip(img_tasks, msk_tasks)):
                img_patch_file = Path(output_dir) / self.satellite / f"img_{idx}.nc" # type: ignore
                msk_patch_file = Path(output_dir) / "annotation"  / f"msk_{idx}.nc"
                img_patch_file.parent.mkdir(parents=True, exist_ok=True)
                msk_patch_file.parent.mkdir(parents=True, exist_ok=True)
                img_tasks_extended.append(img_task + (padding, "reflectance", img_patch_file))
                msk_tasks_extended.append(msk_task +(padding, "class", msk_patch_file))

            with ProcessPoolExecutor(max_workers=available_workers(reduce_by=5)) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._extract_and_combine_patches, task): task for task in img_tasks_extended}
                log_progress(future_tasks=future_tasks, desc="Creating 4D IMG patches")
            
            with ProcessPoolExecutor(max_workers=available_workers()) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._extract_and_combine_patches, task): task for task in msk_tasks_extended}
                log_progress(future_tasks=future_tasks, desc="Creating 4D MSK patches")
    
        return output_dir

    @staticmethod
    def _extract_and_combine_patches(task):
        """
        Helper function that extracts patches from satellite imagery and combines them into a single xarray dataset.
        
        Parameters:
        - task: A tuple containing the following elements:
          - i, j: The starting row and column indices for patch extraction.
          - patch_size: The size of each patch (assumed square).
          - rasters: A list of raster object from type SatelliteImage or SatelliteImageAnnotation
          - padding: A boolean indicating whether to pad patches that are smaller than `patch_size`.
          - var_name: The variable name to be used in the xarray dataset.
          - patches_dir: The directory where the resulting NetCDF file will be saved.
          
        Returns:
        A dictionary with the keys 'status', 'details', and 'error'. 'status' can be 'success' or 'error',
        'details' provides additional information, and 'error' contains error message if any.
        """
        i, j, patch_size, raster_paths, padding, var_name, patch_file = task
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        try:
            patches = []
            timesteps = []
            transform, crs = None, None
            timestamp_path_pairs = [(pd.to_datetime(path.parts[-2]), path) for path in raster_paths]
            timestamp_path_pairs = sorted(timestamp_path_pairs, key=lambda x: x[0])
            for timestep, raster_path in timestamp_path_pairs:
                with rasterio.open(raster_path) as src:
                    args = [j, i, patch_size, patch_size]
                    window = Window(*args)
                    patch = src.read(window=window, boundless=True, fill_value=0)
                    if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        patch = pad_patch(patch, patch_size)
                    elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        continue
                    if transform is None and crs is None:
                        transform = src.window_transform(window)
                        crs = src.crs
                    patches.append(patch)
                    timesteps.append(timestep)

            data = np.stack(patches) if len(patches) > 1 else np.expand_dims(patches[0], axis=0)
            dataset = create_xarray_dataset(data, timesteps, var_name, transform, crs)
            dataset.to_netcdf(patch_file, format="NETCDF4", engine="h5netcdf")
            
            result["status"] = "success"
            result["details"] = f"Successfully created: {patch_file}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc() 
        return result

    def clean(self, resampled_images, indizes, stacked_image, mask):
        """
        Cleans up the satellite data cube by removing unnecessary directories and files.
        
        This method identifies directories and files that are no longer needed, such as 
        patches, resampled images, NDVI and NDWI files, and temporary netCDF files. 
        It then uses a ThreadPoolExecutor to remove these directories and files in parallel 
        to improve the efficiency of the cleanup process.
        """
        # Collect all directories and files to be removed
        files_to_remove = []
        for image in self.images:
            if resampled_images:
                files_to_remove.extend(image.folder.glob("*resampled*"))
            if stacked_image:
                files_to_remove.append(image.path)
            if mask:
                files_to_remove.extend(image.folder.glob("*mask*"))
            if indizes:
                files_to_remove.extend(image.folder.glob("*NDVI*"))
                files_to_remove.extend(image.folder.glob("*NDWI*"))

        if files_to_remove:
            # Use ThreadPoolExecutor to remove directories and files in parallel
            with ThreadPoolExecutor(max_workers=available_workers()) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._safe_delete, file_path): file_path for file_path in files_to_remove if file_path.exists()}
                log_progress(future_tasks, desc="Cleaning Datacube")
        else:
            print("No files to delete")

    @staticmethod
    def _safe_delete(path):
        """
        Safely deletes a given path, whether it's a file or a directory.
        
        This method attempts to delete the specified path. If the path is a directory, it is
        removed with all its contents. If it's a file, it's directly deleted. The method
        handles exceptions gracefully, returning a status message indicating the outcome.
        
        Parameters:
        - path (Path): A pathlib.Path object representing the file or directory to be deleted.
        
        Returns:
        - dict: A dictionary containing the status of the operation ('success' or 'error'),
                details of the operation, and any error message if an error occurred.
        """
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            result["status"] = "success"
            result["details"] = f"Deleted {path}"
        except Exception as e:
            # If an exception occurs, capture its details and mark the status as error
            result["status"] = "error"
            result["error"] = str(e)  
            result["traceback"] = traceback.format_exc() 
        return result 
    
    def plot_spectral_signature(self, spectral_signature, output_folder=""):
        bands = list(spectral_signature[list(spectral_signature.keys())[0]].keys())
        fig, ax = plt.subplots(figsize=(10,6))
        for band in bands:
            time_steps = list(spectral_signature.keys())
            band_values = [spectral_signature[time_step][band] for time_step in time_steps]
            ax.plot(time_steps, band_values, label=band)

        ax.set_title("Spectral Signature over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Band Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        if output_folder:
            plt.savefig(os.path.join(output_folder, f"{self.satellite}_spectralSig_ts{len(time_steps)}.png"))
        return
 
class Sentinel2DataCube(SatelliteDataCube):
    """
    A class for managing Sentinel-2 satellite image data.
    Attributes:
        base_folder (Path): The base directory for the satellite data.
        satellite (str): Name of the satellite, defaulting to 'sentinel-2'.
        satellite_images_folder (Path): Directory containing satellite images.
        ann_file (Path): Path to the primary annotation file.
        images_by_date (dict): Dictionary mapping dates to Sentinel2 images.
    """
    def __init__(self, base_folder):
        """
        Initializes a Sentinel2DataCube with a specified base folder.
        
        Parameters:
            base_folder (str): The path to the base directory where the satellite data is stored.
        """
        super().__init__()
        self.base_folder = Path(base_folder)
        if not self.base_folder.is_dir():
            raise FileNotFoundError(f"Base folder {self.base_folder} does not exist or is not a directory.")

        self.satellite = "sentinel-2"
        self.satellite_images_folder = self.base_folder / self.satellite
        if not self.satellite_images_folder.is_dir():
            raise FileNotFoundError(f"Satellite images folder {self.satellite_images_folder} does not exist or is not a directory.")

        self.ann_file = self._init_annotation()
        self.images = self._init_images()
        self._print_initialization_info()

    def _init_annotation(self):
        """
        Initializes the annotation file from the annotations directory.
        
        Returns:
            Path: The path to the first .shp file found in the annotations directory.
        """
        ann_dir = self.base_folder / "annotations"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory {ann_dir} does not exist or is not a directory.")

        ann_file = next(ann_dir.glob("*.shp"), None)
        if ann_file is None:
            raise FileNotFoundError("No .shp files found in the annotations directory.")
        return ann_file

    def _init_images(self):
        """
        Initializes satellite images from the satellite images folder, organizing them chronologically in a list.

        Returns:
        list: A list of Sentinel2 image instances, sorted by date.
        """
        images = []
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                try:
                    image_instance = Sentinel2(folder=satellite_image_folder)
                    images.append(image_instance)
                except ValueError:
                    logging.warning(f"Skipping {satellite_image_folder.name}: Does not match date format YYYYMMDD.")

        # Sort the list by date
         
        return images

    def add_image(self, image_folder, date):
        """
        Adds a new satellite image to the data cube.

        This method allows the addition of a new Sentinel2 image to the images_by_date dictionary,
        mapping it to a specific date. It assumes the existence of a Sentinel2 class that initializes
        with the image's folder path.

        Parameters:
        - image_folder (str): The folder path where the satellite image is stored.
        - date (datetime.date): The date associated with the satellite image.

        Returns:
        - None
        """
        self.images.append(Sentinel2(image_folder))
        self.images.sort(key=lambda image: image.date)

    def remove_image_by_index(self, idx):
        """
        Removes a satellite image from the data cube by index.

        Parameters:
        - idx (int): The index of the image to be removed in the images list.

        Returns:
        - bool: True if the image was successfully removed, False if the index was out of bounds.
        """
        if 0 <= idx < len(self.images):
            del self.images[idx]
            return True
        return False

    def calculate_ndvis_for_dating(self, output_dir=None, good_pixel_threshold=70):
        output_dir = Path(self.base_folder) / "other" if not output_dir else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        output_path = output_dir / f"{Path(self.base_folder).name}_ann_ndvi_data.json"
        
        if output_path.exists():
            return self.load_spectral_signature(output_path)

        tasks = [(image, self.ann_file, good_pixel_threshold) for image in self.images]
        dc_ann_ndvi_data = {}

        for task in tqdm(tasks, total=len(tasks), desc="Calculate NDVI values for dating"):
            img_date, img_ann_ndvi_data, logs = self._calculate_ndvis_of_image(task)
            for ann_id, ndvis in img_ann_ndvi_data.items():
                dc_ann_ndvi_data.setdefault(ann_id, {})[img_date] = ndvis
            if logs.get("status") != "success":
                print(f"Task raised the following error: {logs.get('error', 'Unknown error')}")

        # with ProcessPoolExecutor(max_workers=available_workers(reduce_by=10)) as executor:
        #     future_tasks = {executor.submit(Sentinel2DataCube._calculate_ndvis_of_image, task): task for task in tasks}
            
        # # Process results outside of the with block
        # for future in tqdm(as_completed(future_tasks), total=len(tasks), desc="Calculating NDVI around annotations"):
        #     try:
        #         img_date, img_ann_ndvi_data, logs = future.result()
        #         for ann_id, ndvis in img_ann_ndvi_data.items():
        #             dc_ann_ndvi_data.setdefault(ann_id, {})[img_date] = ndvis
        #         if logs.get("status") != "success":
        #             print(f"Task raised the following error: {logs.get('error', 'Unknown error')}")
        #     except Exception as exc:
        #         print(f"Unexpected exception {exc}")

        self.save_spectral_signature(dc_ann_ndvi_data, output_path)
        return dc_ann_ndvi_data

    @staticmethod
    def _calculate_ndvis_of_image(task):
        image, ann_file, good_pixel_threshold = task
        result = {"status": "", "details": "", "error": "", "traceback": ""}
        try:
            img_date = image.date.strftime("%Y%m%d")
            ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            img_ann_ndvi_data = ann.calculate_ndvis_for_dating(good_pixel_threshold)
            result["status"] = "success"
            result["details"] = f"Successfully calculated ndvi around annotations for {image} on {img_date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc() 
        return img_date, img_ann_ndvi_data, result
    
    @staticmethod
    def save_spectral_signature(data, path):
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_spectral_signature(path):
        with open(path, 'r') as f:
            return json.load(f)

class Sentinel1DataCube(SatelliteDataCube):

    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images = self._init_images()
        self._print_initialization_info()

    def _init_images(self):
        s1_images = []
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                s1_image = Sentinel1(folder=satellite_image_folder)
                s1_images.append(s1_image)
        return s1_images

    def add_image(self, image_folder):
        self.images.append(Sentinel2(image_folder))
        self.images.sort(key=lambda image: image.date)
        pass

    def stack_bands(self, reset=False):
        for image in self.images:
            if reset:
                if image.path.exists():
                    image.path.unlink()
            image.stack_bands()
            
class Sentinel12Datacube:
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)
        self.s1_datacube = Sentinel1DataCube(base_folder)
        self.s2_datacube = Sentinel2DataCube(base_folder)
        self.annotations = SatCubeAnnotation(base_folder)
        self.patches_dir = self.base_folder / "patches" # Path to the patches directory - store here patches of S1, S2 and Annotations
        self.folds_dir = None
    
    def stack_bands_of_images(self, resolution, include_indizes):
        self.s1_datacube.stack_bands_of_images(resolution, include_indizes)
        self.s2_datacube.stack_bands_of_images(resolution, include_indizes)
        return
    
    def rasterize_annotations(self, resolution):
        self.annotations.rasterize(resolution)
        return

    def create_patches(self, patch_size, total_images, overlay=0, padding=True, output_dir=None):
        if not self.patches_dir.exists():
            self.patches_dir.mkdir(parents=True, exist_ok=True)
            self.s1_datacube.create_patches(patch_size, total_images, overlay, padding, output_dir)
            self.s2_datacube.create_patches(patch_size, total_images, overlay, padding, output_dir)
            self.annotations.create_patches(patch_size, overlay, padding, output_dir)
        return self.patches_dir
    
    def build_folds(self, fold_nr):
        pass

    def calculate_normalization_values(self):
        pass

    def train_model(self, config_file):
        pass

    def clean(self, resampled_images=True, indizes=True, stacked_image=False):
        self.s1_datacube.clean(indizes, stacked_image)
        self.s2_datacube.clean(resampled_images, indizes, stacked_image)
        return