import os
from matplotlib import pyplot as plt
from .image import Sentinel1, Sentinel2
from .annotation import SatelliteImageAnnotation
from .utils import transform_spectal_signature, extract_patch_coordinates, log_progress
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import shutil
import rioxarray
import xarray as xr
from tqdm import tqdm
import numpy as np

class SatelliteDataCube:
    def __init__(self):
        self.base_folder = None
        self.satellite = None
        self.satellite_images_folder = None
        self.ann_file = None
        self.images_by_date = {}
    
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

    def collect_patch_paths(self, process_type):
        """
        Collects paths to image or mask patch files, or both, organized by patch file name across time steps.
        
        Depending on the specified process_type, this function aggregates the file paths for image patches,
        mask patches, or both, within the directory structure of satellite images. The paths are organized
        in a dictionary (or dictionaries, for 'BOTH') with patch names as keys and lists of paths as values.

        Args:
            process_type (str): Type of patches to collect. Accepts 'IMG' for image patches 
                                and 'MSK' for mask patches.
        
        Returns:
            dict or tuple of dicts: For 'IMG' or 'MSK', returns a dictionary with patch names as keys and list of paths as values.
        """
        if process_type not in ["IMG", "MSK"]:
            raise ValueError("Invalid process_type. Must be 'IMG' or 'MSK'.")

        # Initialize dictionaries to hold paths
        image_patches_dict = defaultdict(list)
        mask_patches_dict = defaultdict(list)

        for image_dir in self.satellite_images_folder.iterdir():
            if image_dir.is_dir():
                # Collect image patches
                if process_type == "IMG":
                    img_patches_dir = image_dir / "patches" / process_type
                    if img_patches_dir.exists():
                        for patch_path in img_patches_dir.glob('*.tif'):
                            patch_name = patch_path.stem  # More reliable than split on name
                            image_patches_dict[patch_name].append(patch_path)
                # Collect mask patches
                elif process_type == "MSK":
                    msk_patches_dir = image_dir / "patches" / process_type
                    for patch_path in msk_patches_dir.glob('*.tif'):
                        patch_name = patch_path.stem
                        mask_patches_dict[patch_name].append(patch_path)

        # Return the appropriate data structure based on process_type
        if process_type == "IMG":
            return dict(image_patches_dict)
        elif process_type == "MSK":
            return dict(mask_patches_dict)

    def clean(self):
        # Collect all directories and files to be removed
        dirs_to_remove = []
        files_to_remove = []
        
        for image in self.images_by_date.values():
            dirs_to_remove.append(image.folder / "patches")
            files_to_remove.extend(image.folder.glob("*resampled*"))
            files_to_remove.extend(image.folder.glob("*NDVI*"))
            files_to_remove.extend(image.folder.glob("*NDWI*"))
            files_to_remove.extend(image.folder.glob("*.tmp.nc"))

        # Use ThreadPoolExecutor to remove directories and files in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: 
            future_tasks_dirs = {executor.submit(SatelliteDataCube.safe_delete, dir_path): dir_path for dir_path in dirs_to_remove if dir_path.exists()}
            future_tasks_files = {executor.submit(SatelliteDataCube.safe_delete, file_path): file_path for file_path in files_to_remove if file_path.exists()}
            # Combine the future tasks for directories and files
            future_tasks = {**future_tasks_dirs, **future_tasks_files}
            log_progress(future_tasks, desc="Cleaning Datacube")
    
    @staticmethod
    def safe_delete(path):
        result = {"status": "", "details": "", "error": ""}
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)  # Consider `ignore_errors=True` for a more robust deletion.
            else:
                path.unlink()
            result["status"] = "success"
            result["details"] = f"Deleted {path}"
            return result
        except Exception as e:
            result["status"] = "error",
            result["error"] = str(e)  # Provides the error message if an error occurred.
            return result

    def load_image(self, date):
        if date in self.images_by_date:
            return self.images_by_date[date]
        else:
            raise ValueError(f"Date is not in SatelliteDataCube. Please use the function find_closest_date to use a valid date.")

    def find_closest_date(self, date):
        satellite_images_dates = sorted(self.images_by_date.keys()) # list of dates
        date_idx = satellite_images_dates.index(date)
        next_date = satellite_images_dates[date_idx+1]
        previous_date = satellite_images_dates[date_idx-1]
        if abs(date - next_date) >= abs(date - previous_date):
            return previous_date
        else:
            return next_date

    def select_images_with_number(self, number_of_images):
        satellite_image_dates = [satellite_image.date for satellite_image in self.images_by_date.keys()]
        interval = len(satellite_image_dates) // number_of_images
        selected_dates = [satellite_image_dates[i * interval] for i in range(number_of_images)]
        self.selected_images_by_date = {date: self.images_by_date[date] for date in selected_dates if date in self.images_by_date}
        return 
    
    def select_images_per_month(self, number_of_images_per_month):
        # Extract start and end year
        min_year = min(self.images_by_date.keys()).year
        max_year = max(self.images_by_date.keys()).year
        
        selected_images_per_month = {}
        for year in range(min_year, max_year, 1):
            for month in range(1,12):
                images_month = {date:image for date, image in self.images_by_date.items() if date.year == year and date.month == month}
                images_month_badPixeRatio = {date: image.calculate_bad_pixel_ratio() for date, image in images_month.items()}
                selected_images_pre_month = {date: self.images_by_date[date] for date, badPixelRatio in sorted(images_month_badPixeRatio.items(), key=lambda item: item[1])[:number_of_images_per_month]}
        self.selected_images_by_date = selected_images_per_month
        return self.selected_images_by_date
    
    def select_images_with_dates(self, dates):
        self.selected_images_by_date = {date: self.images_by_date[date] for date in dates if date in self.images_by_date}
        return self.selected_images_by_date

    def create_spectral_signature_of_single_annotation(self, annotation_id, indizes=False):
        ann_dc_spectral_sig = {}
        for image_date, image in self.selected_images_by_date.items():
            ann_spetral_signature = image.create_spectral_signature_of_single_annotation(annotation_id, indizes=indizes)
            ann_dc_spectral_sig[image_date] = ann_spetral_signature
        return ann_dc_spectral_sig

    def create_spectral_signature_of_all_annotations(self, indizes=False):
        datacube_spectral_sig = {}
        for image_date, image in self.selected_images_by_date.items():
            annotations_speSignature = image.create_spectral_signature(annotation=self.annotation, indizes=indizes)
            datacube_spectral_sig[image_date] = annotations_speSignature
        self.spectral_signature = transform_spectal_signature(datacube_spectral_sig) # self.spectral_signature['B02'])
        return self.spectral_signature
   
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
   
    def create_msk_patches(self, patchsize, overlay):
        tasks = [(image, patchsize, overlay, self.ann_file) for image in self.images_by_date.values()]

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._process_mask_patch, task): task for task in tasks}
        
            count = 0
            for future in as_completed(future_tasks):
                count += 1  # Increment the counter for each completed task
                image = future_tasks[future][0]
                try:
                    future.result()
                    if count % 10 == 0:  # Check if the count is divisible by 10
                        print(f"10 image masks processed. Most recent: {image} on {image.date}")
                except Exception as exc:
                    print(f"Task generated an exception: {image.date}, {exc}")    
        
    def create_img_patches(self, patchsize, overlay):
        tasks = [(image, patchsize, overlay) for image in self.images_by_date.values()]
        # Use ProcessPoolExecutor to parallelize the task
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._process_image_patch, task): task for task in tasks}
                
            count = 0
            for future in as_completed(future_tasks):
                count += 1  # Increment the counter for each completed task
                image = future_tasks[future][0]
                try:
                    future.result()
                    if count % 10 == 0:  # Check if the count is divisible by 10
                        print(f"10 images processed. Most recent: {image} on {image.date}")
                except Exception as exc:
                    print(f"Task generated an exception: {image.date}, {exc}")          

    @staticmethod
    def _process_mask_patch(task):
        """
        Process a single mask patch task. This function is designed to be run in parallel.
        """
        image, patchsize, overlay, ann_file = task
        patches_dir = image.folder / "patches" / "MSK"
        result = {"path": str(patches_dir), "status": "", "details": "", "error": ""}
        
        # Check if patches already exist to avoid redundant processing
        if patches_dir.exists():
            result["status"] = "skipped"
            result["details"] = f"MSK patches already exist for {image} on {image.date}. Skipping patch creation..."
            return result
        try:
            img_ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=ann_file)
            img_ann.rasterize(resolution=10)
            img_ann.create_patches(patch_size=patchsize, overlay=overlay)
            result["status"] = "success"
            result["details"] = f"Successfully processed MSK patches for {image} on {image.date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    @staticmethod
    def _process_image_patch(task):
        """
        Process a single image patch task. This function is designed to be run in parallel.
        """
        image, patchsize, overlay = task
        patches_dir = image.folder / "patches" / "IMG"
        result = {"path": str(patches_dir), "status": "", "details": "", "error": ""}
        
        # Check if patches already exist to avoid redundant processing
        if patches_dir.exists():
            result["status"] = "skipped"
            result["details"] = f"IMG patches already exist for {image} on {image.date}. Skipping patch creation..."
            return result
        try:
            image.stack_bands()  # Prepare image by stacking bands
            image.create_patches(patchsize, overlay=overlay)  # Create and store patches
            result["status"] = "success"
            result["details"] = f"Successfully processed IMG patches for {image} on {image.date}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    @staticmethod
    def _process_patch_timeseries(task):
        """
        Processes a single patch to create a time series and saves it as a netCDF file.

        Parameters:
            patch_id (str): Identifier for the patch.
            patch_paths (list): List of file paths to the patch files.
            var_name (str): Variable name to use in the netCDF dataset.
            patches_folder (Path): The directory where the netCDF file should be saved.
        """
        patch_id, patch_paths, var_name, patches_folder = task
        result = {"status": "", "details": "", "error": ""}
        try:
            patches_xarrays = [rioxarray.open_rasterio(path) for path in patch_paths]
            patch_ts = xr.concat(patches_xarrays, dim='time').to_dataset(name=var_name)
            patch_ts_filepath = patches_folder / f"{patch_id}.nc"
            patch_ts.to_netcdf(patch_ts_filepath, format="NETCDF4", engine="h5netcdf")
            patch_ts.close()
            result["status"] = "success"
            result["details"] = f"Successfully processed patch with ID {patch_id}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            if patches_xarrays:
                for da in patches_xarrays:
                    try:
                        da.close()
                    except:
                        pass  # If close method doesn't exist or fails, continue

        return result
    
    def build_patch_timeseries(self, process_type):
        # Validate process_type and retrieve patches
        if process_type not in ["IMG", "MSK"]:
            raise ValueError(f"Invalid process_type: {process_type}. Expected one of ['IMG', 'MSK'].")
        var_name = {"IMG": "reflectance", "MSK": "class"}.get(process_type)
        dir_name = self.satellite if process_type == "IMG" else "annotation"

        patches = self.collect_patch_paths(process_type=process_type)
        patches_folder = self.base_folder / "patches" / dir_name
        patches_folder.mkdir(parents=True, exist_ok=True)

        # Prepare tasks for parallel processing
        tasks = [(patch_id, patch_paths, var_name, patches_folder) for patch_id, patch_paths in patches.items()]

        # Parallel processing
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = {executor.submit(self._process_patch_timeseries, task): task for task in tasks}
            log_progress(future_tasks, desc="Building 4D patches")

    def reduce_msk_patch_timeseries(self, timestep):
        """
        Reduces the time series of mask patches to a single specified timestep.
        """
        patches_folder = self.base_folder / "patches" / "annotation"
        tasks = [(patch_path, timestep) for patch_path in patches_folder.iterdir() if patch_path.suffix == '.nc']

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._cut_timeseries, task): task for task in tasks}
            log_progress(future_tasks, desc="Reduce MSK patch timeseries")

    @staticmethod
    def _cut_timeseries(task):
        patch_path, timestep = task
        result = {"status": "", "details": "", "error": ""}
        try:
            with xr.open_dataset(patch_path) as msk_data:
                reduced_ds = msk_data.isel(time=timestep)
                encoding = {var: {'_FillValue': None} for var in reduced_ds.data_vars}
                # Temporary file ensures that we do not lose data if the process is interrupted
                temp_path = patch_path.with_suffix('.tmp.nc')
                reduced_ds.to_netcdf(temp_path, mode='w', encoding=encoding)
                temp_path.rename(patch_path)  # Atomically replace the old file
                result["status"] = "success"
                result["details"] = f"Successfully reduced MSK patch with ID {patch_path.name}."
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return result

    @staticmethod
    def _is_patch_annotated(task):
        """Checks if a mask patch contains annotations and returns its classification."""
        img_path, msk_path = task
        with xr.open_dataset(msk_path) as msk_data:  # Automatically closes the dataset
            ann_count = np.count_nonzero(msk_data['class'].values == 1)
        return ann_count > 10, task

    def classify_patches(self):
        """Categorizes patches into annotated and non-annotated based on mask annotations."""
        img_patches_dir = self.base_folder / "patches" / self.satellite
        msk_patches_dir = self.base_folder / "patches" / "annotation"
        img_files = sorted(img_patches_dir.iterdir(), key=extract_patch_coordinates)
        msk_files = sorted(msk_patches_dir.iterdir(), key=extract_patch_coordinates)
        patch_pairs = list(zip(img_files, msk_files))

        categorized_patches = {"annotated": [], "non_annotated": []}
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = {executor.submit(SatelliteDataCube._is_patch_annotated, pair): pair for pair in patch_pairs}

            with tqdm(total=len(future_tasks), desc="Categorize patches") as pbar:
                for future in as_completed(future_tasks):
                    pair = None  # Initialize pair to None or a default value
                    try:
                        has_annotation, pair = future.result()
                        category = "annotated" if has_annotation else "non_annotated"
                        categorized_patches[category].append(pair)
                    except Exception as exc:
                        tqdm.write(f"Error processing task. Exception: {exc}")
                    finally:
                        pbar.update(1)
                        pbar.refresh()
        return categorized_patches
    
class Sentinel2DataCube(SatelliteDataCube):

    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.ann_file = self._init_annotation()
        self.images_by_date = self._load_satellite_images()
        self._print_initialization_info()

    def _init_annotation(self):
        ann_dir = self.base_folder / "annotations" 
        ann_file = [file for file in ann_dir.glob("*.shp")][0]
        return ann_file

    def _load_satellite_images(self):
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                # annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
                images_by_date[date_satellite_image] = Sentinel2(satellite_image_folder)
        satellite_images_by_date_sorted = dict(sorted(images_by_date.items()))
        return satellite_images_by_date_sorted

    def _find_higher_quality_satellite_image(self, satellite_image, search_limit=5):
        """Search for the nearest good quality image before and after the current date. 
        If none is found, return the one with the least bad pixels from the search range."""
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

class Sentinel1DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1-nrb"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images_by_date = self._load_satellite_images()
        self._print_initialization_info()
     
    def _load_annotation(self):
        annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
        return SatelliteImageAnnotation(annotation_shapefile)
    
    def _load_satellite_images(self):
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
                images_by_date[date_satellite_image] = Sentinel1(satellite_image_folder, annotation_shapefile, date_satellite_image)
        satellite_images_by_date_sorted = dict(sorted(images_by_date.items()))
        return satellite_images_by_date_sorted

    def _find_higher_quality_satellite_image(self, satellite_image, search_limit=5):
        pass

    def find_best_selected_images(self):
        pass
           
class Sentinel12DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-fusion"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images_by_date = self._load_satellite_images()
        self._print_initialization_info()