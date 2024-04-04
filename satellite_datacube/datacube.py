import os
from matplotlib import pyplot as plt
from .image import Sentinel1, Sentinel2
from .annotation import SatelliteImageAnnotation
from .utils import save_spectral_signature, load_spectral_signature, pad_patch, log_progress, create_xarray_dataset, available_workers
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
from collections import OrderedDict

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
        self.images_by_date = {}
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
        #print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print(f"{2*divider}")
        print("Initialized data-cube with following parameter:")
        print(f"- Base folder: {self.base_folder}")
        print(f"- Satellite mission: {self.satellite}")
        print(f"- Start-End: {min(self.images_by_date.keys())} -> {max(self.images_by_date.keys())}")
        print(f"- Length of data-cube: {len(self.images_by_date)}")
        print(f"{2*divider}")
    
    def select_images_for_timeseries(self, total_images=80):
        """
        Selects images to fill a timeseries of a specific length by randomly adding images 
        to reach the desired total after allocating a base number per month.
        """
        # Determine the number of months with data
        unique_months = {(date.year, date.month) for date in self.images_by_date.keys() }
        months_count = len(unique_months)
        base_number_per_month = int(total_images/months_count)

        images = list(self.images_by_date.items())
        with ProcessPoolExecutor(max_workers=available_workers()) as executor:
            results = list(tqdm(executor.map(SatelliteDataCube._calculate_bad_pixel_ratio_for_image, images), total=len(images), desc="Calculate bad pixel ratio of images"))

        ratios = {image[0]: result for image, result in zip(images, results)}
        all_images_sorted = sorted(images, key=lambda item: ratios[item[0]])

        selected_images_per_month = {}
        for date, image in all_images_sorted:
            year_month = (date.year, date.month)
            if len(selected_images_per_month.get(year_month, [])) < base_number_per_month:
                selected_images_per_month.setdefault(year_month, []).append(image)

        # Randomly select remainder images
        all_images_flat = [image for _, image in all_images_sorted]
        selected_images = [image for images in selected_images_per_month.values() for image in images]
        additional_images = [image for image in all_images_flat if image not in selected_images]
        additional_selected_images = random.sample(additional_images, total_images-len(selected_images))

        # Add additional images randomly
        for image in additional_selected_images:
            year_month = (image.date.year, image.date.month)
            selected_images_per_month[year_month].append(image)

        selected_images_per_month = OrderedDict(sorted(selected_images_per_month.items(), key=lambda item:item[0]))

        return selected_images_per_month
    
    @staticmethod
    def _calculate_bad_pixel_ratio_for_image(item):
        key, image = item
        return key, image.calculate_bad_pixel_ratio()

    def select_images_with_dates(self, dates):
        """
        Selects and returns satellite images that match a given list of dates.
        
        This method filters the images stored in the data cube by the specified dates. It updates
        the `selected_images_by_date` attribute with a dictionary of images that match the given
        dates. If a date in the list does not correspond to any image in the data cube, it is 
        simply ignored.
        
        Parameters:
        - dates (list of datetime.date): A list of dates for which to select matching satellite images.
        
        Returns:
        - dict: A dictionary mapping the selected dates to their corresponding satellite images. 
                The keys are dates, and the values are the satellite images (instances of Sentinel2
                or another relevant class).
        """
        # Filter the `images_by_date` dictionary to include only the images with dates present in the given list.
        selected_images_by_date = {date: self.images_by_date[date] for date in dates if date in self.images_by_date}

        return selected_images_by_date

    def get_image_dates_range(self):
        start_date = min(self.images_by_date.keys())
        end_date = max(self.images_by_date.keys())
        return start_date, end_date
        
    def create_spectral_signature(self, band_ids, output_dir=None, filtering=False):
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
        output_path = output_dir / f"{self.base_folder.name}_S2_specSig.json" 
        if not output_path.exists():
            print(f"Start extracting values for building a spectral signature with filtering: {filtering}")
            tasks = [(image, self.ann_file, band_ids, filtering) for image in self.images_by_date.values()]
            with ProcessPoolExecutor() as executor:
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
        image, ann_file, band_ids, filtering = task
        result = {"status": "", "details": "", "error": ""}
        try:
            img_date = image.date.strftime("%Y%m%d")
            ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            img_specSig = ann.create_spectral_signature(band_ids, filtering) 
            result["status"] = "success"
            result["details"] = f"Successfully created spectral signature for {image} on {img_date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return img_date, img_specSig, result
            
    def rasterize_annotations(self, resolution):
        tasks = [(image, self.ann_file, resolution) for image in self.images_by_date.values()]
        with ProcessPoolExecutor(max_workers=available_workers()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._rasterize_annotation, task): task for task in tasks}
            log_progress(future_tasks=future_tasks, desc="Rasterize annotation of images")
        return 
    
    @staticmethod
    def _rasterize_annotation(task):
        image, ann_file, resolution = task
        result = {"status": "", "details": "", "error": ""}
        try:
            annotation = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            annotation.rasterize(resolution=resolution)
            result["status"] = "success"
            result["details"] = f"Successfully processed IMG patches for {image} on {image.date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def stack_bands_of_images(self, resolution=10, include_indizes=False):
        tasks = [(image, resolution, include_indizes) for image in self.images_by_date.values()] # necessary to put it in tuple - for serialization 

        with ProcessPoolExecutor(max_workers=available_workers()-5) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._stacking_image_bands, task): task for task in tasks}
            log_progress(future_tasks=future_tasks, desc="Stacking bands of images")
        return list(self.images_by_date.values())
    
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
      
    def create_img_patches(self, patch_size, overlay=0, padding=True, output_dir=None, include_indizes=False):
        """
        Splits satellite imagery into smaller, overlapped patches for processing or analysis, and saves them to a specified directory.

        This method divides a set of satellite images into patches of a specified size, optionally with an overlay between adjacent patches and padding at the image edges. It is designed to handle large satellite data cubes by processing images in parallel, improving efficiency for large-scale analysis.

        Parameters:
        - patch_size (int): The size of each square patch (in pixels). Each patch will be patch_size x patch_size.
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
        output_dir = output_dir  if output_dir else self.base_folder 
        patches_dir = Path(output_dir) / self.satellite # type: ignore
        if not patches_dir.exists():
            var_name = "reflectance" 
            tasks = []
            
            if not include_indizes:
                images_paths = [image.path for image in self.images_by_date.values()]
            else:
                images_paths = [image.index_path for image in self.images_by_date.values()]

            # Filter/Pad images
            selected_images_per_month = self.select_images_for_timeseries(total_images=80)
            print(f"Length of selected image timeseries of datacube: {len([image.date for images in selected_images_per_month.values() for image in images])}")
            #[print(f"{date}: {len(images)}") for date, images in selected_images_per_month.items()]
                        
            with rasterio.open(images_paths[0]) as src:
                step_size = patch_size - overlay
                for i in range(0, src.width, step_size):
                    for j in range(0, src.height, step_size):
                        task = (i, j, patch_size, images_paths, padding, var_name, patches_dir)
                        tasks.append(task)
            
            print(tasks[0])
            self._extract_and_combine_patches(tasks[0])

            import pdb
            pdb.set_trace()
            
            with ProcessPoolExecutor(max_workers=available_workers()) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._extract_and_combine_patches, task): task for task in tasks}
                log_progress(future_tasks=future_tasks, desc="Creating 4D IMG patches")
        return patches_dir

    def create_msk_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        """
        Splits annotations of satellite imagery into smaller, overlapped patches for processing or analysis, and saves them to a specified directory.

        This method divides a set of satellite image annotations into patches of a specified size, optionally with an overlay between adjacent patches and padding at the image edges. It is designed to handle large satellite data cubes by processing images in parallel, improving efficiency for large-scale analysis.

        Parameters:
        - patch_size (int): The size of each square patch (in pixels). Each patch will be patch_size x patch_size.
        - overlay (int, optional): The overlap size (in pixels) between adjacent patches. Defaults to 0, meaning no overlap.
        - padding (bool, optional): If True, adds padding to the patches at the edges of the images. Defaults to True.
        - output_dir (Path or str, optional): The directory where the patches will be saved. If not specified, patches will be saved in a default directory structure: base_folder/patches/satellite.

        The method first gathers paths of satellite images to process, calculates the number of patches based on the specified patch size and overlay, and then processes each patch in parallel, utilizing all available CPU cores. Each patch is extracted from the satellite images annotation, optionally padded, and saved to the output directory in a 4D mask format suitable for further analysis.

        Returns:
        None. The patches are saved directly to the specified output directory.

        Examples:
        # Basic usage with default settings
        SatelliteDataCube.create_msk_patches(patch_size=256)

        # Custom usage with an overlay of 10 pixels, padding disabled, and specifying an output directory
        SatelliteDataCube.create_msk_patches(patch_size=512, overlay=10, padding=False, output_dir="path/to/output/dir")
        """
        output_dir = output_dir if output_dir else self.base_folder
        patches_dir = Path(output_dir) / "annotation"
        if not patches_dir.exists():
            var_name = "class" 
            # annotations = self.rasterize_annotations(resolution=10)
            masks_paths = [SatelliteImageAnnotation(image, self.ann_file).mask_path for image in self.images_by_date.values()]            
            mask_path = masks_paths[0] # we use a global mask that is equal for all timesteps, so we need only one timestep - saving computation and memory
            tasks = []

            with rasterio.open(mask_path) as src:
                step_size = patch_size - overlay
                for i in range(0, src.width, step_size):
                    for j in range(0, src.height, step_size):
                        task = (i, j, patch_size, [mask_path], padding, var_name, patches_dir)
                        tasks.append(task)

            with ProcessPoolExecutor(max_workers=available_workers()) as executor:
                future_tasks = {executor.submit(SatelliteDataCube._extract_and_combine_patches, task): task for task in tasks}
                log_progress(future_tasks=future_tasks, desc="Creating 4D MSK patches")
        
        return patches_dir

    @staticmethod
    def _extract_and_combine_patches(task):
        """
        Helper function that extracts patches from satellite imagery and combines them into a single xarray dataset.
        
        Parameters:
        - task: A tuple containing the following elements:
          - i, j: The starting row and column indices for patch extraction.
          - patch_size: The size of each patch (assumed square).
          - raster_paths: A list of paths to the raster files from which patches will be extracted.
          - padding: A boolean indicating whether to pad patches that are smaller than `patch_size`.
          - var_name: The variable name to be used in the xarray dataset.
          - patches_dir: The directory where the resulting NetCDF file will be saved.
          
        Returns:
        A dictionary with the keys 'status', 'details', and 'error'. 'status' can be 'success' or 'error',
        'details' provides additional information, and 'error' contains error message if any.
        """
        i, j, patch_size, raster_paths, padding, var_name, patches_dir = task
        result = {"status": "", "details": "", "error": ""}
        try:
            patches = []
            transform, crs = None, None
            timesteps = [pd.to_datetime(path.parts[-2]) for path in raster_paths]
            for raster_path in raster_paths:
                with rasterio.open(raster_path) as src:
                    window = Window(j, i, patch_size, patch_size)
                    patch = src.read(window=window, boundless=True, fill_value=0)
                    if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        patch = pad_patch(patch, patch_size)
                    elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        continue
                    if transform is None and crs is None:
                        transform = src.transform
                        crs = src.crs
                    patches.append(patch)

            data = np.stack(patches) if len(patches) > 1 else np.expand_dims(patches[0], axis=0)
            dataset = create_xarray_dataset(data, timesteps, var_name, transform, crs)

            patch_ts_filepath = patches_dir / f"patch_{i}_{j}.nc"
            patch_ts_filepath.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_netcdf(patch_ts_filepath, format="NETCDF4", engine="h5netcdf")
            
            result["status"] = "success"
            result["details"] = f"Successfully processed patches for patch_{i}_{j}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def clean(self, indizes=False):
        """
        Cleans up the satellite data cube by removing unnecessary directories and files.
        
        This method identifies directories and files that are no longer needed, such as 
        patches, resampled images, NDVI and NDWI files, and temporary netCDF files. 
        It then uses a ThreadPoolExecutor to remove these directories and files in parallel 
        to improve the efficiency of the cleanup process.
        """
        # Collect all directories and files to be removed
        files_to_remove = []
        for image in self.images_by_date.values():
            files_to_remove.extend(image.folder.glob("*resampled*"))
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
        result = {"status": "", "details": "", "error": ""}
        try:
            if path.is_dir():
                # Remove the directory and all its contents, ignoring any errors that occur during deletion
                shutil.rmtree(path, ignore_errors=True)
            else:
                # Remove the file
                path.unlink()
            result["status"] = "success"
            result["details"] = f"Deleted {path}"
        except Exception as e:
            # If an exception occurs, capture its details and mark the status as error
            result["status"] = "error",
            result["error"] = str(e)  # Store the error message
            
        return result  # Return the result dictionary with the operation's outcome
    
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
        self.images_by_date = self._init_images()
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
        Initalizes satellite images from the satellite images folder, organizing them by date.

        Returns:
            dict: A dictionary mapping dates to Sentinel2 image instances, sorted by date.
        """
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                try:
                    date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                    images_by_date[date_satellite_image] = Sentinel2(folder=satellite_image_folder)
                except ValueError:
                    logging.warning(f"Skipping {satellite_image_folder.name}: Does not match date format YYYYMMDD.")
        
        return dict(sorted(images_by_date.items()))

    def _find_higher_quality_satellite_image(self, satellite_image, search_limit=5):
        """Search for the nearest good quality image before and after the current date. 
        If none is found, return the one with the least bad pixels from the search range.
        
        Returns:
            Sentinel2: SatelliteImage object with the best quality
        """
        satellite_images_dates = sorted(self.images_by_date.keys())
        start_date_idx = satellite_images_dates.index(satellite_image.date)

        alternative_satellite_images = []
        # Search within the range for acceptable images
        for offset in range(1, search_limit + 1):
            for direction in [-1, 1]:
                new_date_idx = start_date_idx + (direction * offset)
                if 0 <= new_date_idx < len(satellite_images_dates):
                    new_date = satellite_images_dates[new_date_idx]
                    neighbor_satellite_image = self.images_by_date.get(new_date)
                    if neighbor_satellite_image.is_quality_acceptable():
                        return neighbor_satellite_image
                    else:
                        alternative_satellite_images.append((neighbor_satellite_image, neighbor_satellite_image.calculate_bad_pixels()))
                else:
                    continue
        alternative_satellite_images.sort(key=lambda x: x[1])  # Sorting by bad pixel ratio
        return alternative_satellite_images[0][0] if alternative_satellite_images else satellite_image

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
        # Assume Sentinel2 is a class that represents a satellite image and requires
        # the folder path for initialization. Replace Sentinel2 with the correct class name as necessary.
        self.images_by_date[date] = Sentinel2(image_folder)

    def remove_image(self, date):
        """
        Removes a satellite image from the data cube.

        This method removes an image associated with a specific date from the images_by_date dictionary,
        effectively deleting it from the data cube.

        Parameters:
        - date (datetime.date): The date of the image to be removed.

        Returns:
        - bool: True if the image was successfully removed, False if the image was not found.
        """
        # Check if the date exists in the dictionary before attempting to remove it
        if date in self.images_by_date:
            del self.images_by_date[date]
            return True
        else:
            return False

    def find_best_selected_images(self):
        try: 
            updated_images_by_date = {}
            for date, image in self.selected_images_by_date.items():
                print("[" + " ".join(str(x) for x in range(len(self.images_by_date.keys()) + 1)) + "]", end='\r')
                if image.is_quality_acceptable():
                    updated_images_by_date[date] = image
                else:
                    neighbour_satellite_image = self._find_higher_quality_satellite_image(image)
                    updated_images_by_date[neighbour_satellite_image.date] = neighbour_satellite_image
            self.selected_images_by_date = updated_images_by_date
        except ValueError:
            raise ValueError("An error occurred while updating the satellite images of data-cube.Please make sure that you first select images with the functions select_images_by date() or select_imgaes_by_date().")
        return 
        
    def select_images_with_badPixelRatio(self, bad_pixel_ratio):
        selected_dates = []
        for date, image in self.images_by_date.items():
            image_bad_pixel_ratio = image.calculate_bad_pixel_ratio()
            if image_bad_pixel_ratio <= bad_pixel_ratio:
                selected_dates.append(date)
        self.selected_images_by_date = {date: self.images_by_date[date] for date in selected_dates if date in self.images_by_date}
        return
    
    def calculate_ndvi_around_annotations(self, output_dir=None, buffer_distance=10):
        """
        Asynchronously calculates NDVI values around annotations for each image in the dataset,
        and saves the results in a JSON file. If the file already exists, the saved results are loaded instead.

        Parameters:
        - output_dir (Path, optional): The directory where the output JSON file is saved. If not provided,
        a default 'other' directory within the base folder is used.
        - buffer_distance (int, optional): The distance in meters to buffer around each annotation before
        calculating NDVI. Defaults to 10.

        Returns:
        - dict: A dictionary with annotation IDs as keys, each mapping to a dictionary of dates with
        their corresponding NDVI values.
        """
        output_dir = self.base_folder / "other" if not output_dir else Path(output_dir)
        output_path = output_dir / f"{self.base_folder.name}_ndvi_around_ann.json"
        if not output_path.exists():
            tasks = [(image, self.ann_file, buffer_distance) for image in self.images_by_date.values()]
            with ProcessPoolExecutor() as executor:
                future_tasks = {executor.submit(SatelliteDataCube._calculate_ndvi_around_ann_of_image, task): task for task in tasks}
                dc_ndvi_around_ann = {}
                for future in tqdm(as_completed(future_tasks), total=len(future_tasks), desc="Calculation NDVI around annotations"):
                    try:
                        img_date, img_ndvi_around_ann, logs = future.result()
                        for ann_id, bands in img_ndvi_around_ann.items():
                            if ann_id not in dc_ndvi_around_ann:
                                dc_ndvi_around_ann[ann_id] = {}
                            dc_ndvi_around_ann[ann_id][img_date] = bands
                        if logs.get("status") != "success":
                            print(f"Task raised following error: {logs.get('error', 'Unknown error')}")
                    except Exception as exc:
                        print(f"Unexpected exception {exc}: {traceback.format_exc()}")
                save_spectral_signature(dc_ndvi_around_ann, output_path)
                return dc_ndvi_around_ann
        else:
            return load_spectral_signature(output_path)

    @staticmethod
    def _calculate_ndvi_around_ann_of_image(task):
        """
        Helper function to calculate NDVI around annotations for a single image.
        Intended to be used with concurrent execution in mind.

        Parameters:
        - task (tuple): A tuple containing the image, annotation file path, and buffer distance.

        Returns:
        - tuple: Contains the image date, calculated NDVI around annotations, and a result dictionary
        indicating the status of the operation.
        """
        image, ann_file, buffer_distance = task
        result = {"status": "", "details": "", "error": ""}
        try:
            img_date = image.date.strftime("%Y%m%d")
            ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            img_ndvi_around_ann = ann.get_ndvi_around_polygon(buffer_distance)
            result["status"] = "success"
            result["details"] = f"Successfully calculated ndvi around annotations for {image} on {img_date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return img_date, img_ndvi_around_ann, result

class Sentinel1DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1-nrb"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images_by_date = self._load_satellite_images()
        self._print_initialization_info()
     
    def _load_satellite_images(self):
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
                images_by_date[date_satellite_image] = Sentinel1(satellite_image_folder, annotation_shapefile, date_satellite_image)
        satellite_images_by_date_sorted = dict(sorted(images_by_date.items()))
        return satellite_images_by_date_sorted