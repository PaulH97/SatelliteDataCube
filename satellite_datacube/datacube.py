import os
from matplotlib import pyplot as plt
from .image import Sentinel1, Sentinel2
from .annotation import SatelliteImageAnnotation
from .utils import transform_spectal_signature
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import shutil
import rioxarray
import xarray as xr

# TODO: Update the function with storing the shapefile in a varibale -> creating mask with that and not using the annotation.tif?
# maybe i should select timeseries and store it not as instance variable?
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
        elif process_type == "BOTH":
            return (dict(image_patches_dict), dict(mask_patches_dict))

    def clean(self):
        # Removes all files that are not needed anymore like resampled bands, single timestep patches
        for date, image in self.images_by_date.items():
            patches_folder = image.folder / "patches" 
            patches_ann_folder = image.folder / "annotation" / "patches"
            for folder in [patches_folder, patches_ann_folder]:
                if folder.exists():
                    shutil.rmtree(patches_folder)  
                    print(f"Cleaned image folder on date {date} by removing resampled bands and patches.")
            try:
                resampled_files = [file for file in image.folder.iterdir() if file.is_file() and ("resampled" in file.name or "NDVI" in file.name or "NDWI" in file.name)]
                for resampled_file in resampled_files:
                    resampled_file.unlink()
            except OSError as e:
                print(f"Error deleting file {resampled_file}: {e.strerror}")
                            
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

    def create_patches(self, patchsize, overlay, process_type):
        """
        Generates patches for satellite images, masks, or both based on the specified process type.

        This function iterates over images in the data cube, processing them according to the
        selected type: images only ('IMG'), masks only ('MSK'), or both images and masks ('BOTH').
        It utilizes specific patchify functions tailored for handling images and masks to create
        the patches. Patches are stored in the satellite image folder under '.../patches/IMG' or '.../patches/MSK'.

        Parameters:
            patchsize (int): The size of the square patch to be generated. This size applies to
                both dimensions of the patch (width and height) in pixels.
            process_type (str): Specifies the type of data to patchify. Acceptable values are
                'IMG' for images, 'MSK' for masks. The function will execute different patchify 
                functions based on this parameter.
        Raises:
            ValueError: If `process_type` is not one of the expected values ('IMG', 'MSK', 'BOTH'),
                a ValueError is raised to alert the user of the invalid input.
        Note:
            The function expects `self.images_by_date` to be a dictionary where keys are dates
            and values are image objects, and `self.ann_file` to be the path to the annotation
            file used for mask generation when process_type is 'MSK' or 'BOTH'.
        """
        valid_process_types = ["IMG", "MSK"]
        if process_type not in valid_process_types:
            raise ValueError(f"Invalid process_type: {process_type}. Expected one of {valid_process_types}.")

        for date, image in self.images_by_date.items():
            print(f"Process S2 image on date: {date}")
            # Execute the selected patchify function for generating patches of selected data (images, masks or both)
            if process_type == "IMG":
                img_patches_dir = image.folder / "patches" / process_type
                if not img_patches_dir.exists():
                    print(f"--Creating patches of image with patch size of {patchsize}px")
                    image = image.stack_bands()  # stack all bands, scl and msk together
                    image.create_patches(patchsize, overlay=overlay)
                else:
                    print("--Patches already exists for this image")
            else:
                ann_patches_dir = image.folder / "patches" / process_type
                if not ann_patches_dir.exists():
                    print(f"--Creating patches of image annotation with patch size of {patchsize}px")
                    img_ann = SatelliteImageAnnotation(satellite_image=image, shapefile_path=self.ann_file)
                    img_ann.rasterize(resolution=10)
                    img_ann.create_patches(patchsize)
                else:
                    print("--Patches already exists for this image annotation")

    def build_patch_timeseries(self, process_type):
        """
        Constructs a time series for each patch and stores it as a netCDF file.
        
        This function iterates over patches, loads them as xarray DataArrays, and concatenates
        them along a new 'time' dimension to create a 4D time series (time x channels x height x width).
        The resulting time series is then saved as a netCDF file, with the file naming convention
        reflecting the patch type and identifier.

        Parameters:
            patches (dict): A dictionary where keys are patch identifiers and values are lists of
                file paths to the individual patch files.
            process_type (str): Specifies the type of data to patchify. Acceptable values are
                'IMG' for images, 'MSK' for masks.
        Note:
            The function assumes the existence of `self.base_folder`, which specifies the base directory
            where the netCDF files should be stored. The directory structure is organized by patch type.
        """
        print("Start building 4D xarrays patches and store them as netCDF files (shape: time x channels x height x width):")

        # Determine the variable name based on the patch type
        var_name = {"IMG": "reflectance", "MSK": "class"}.get(process_type, "data")

        valid_process_types = ["IMG", "MSK"]
        if process_type not in valid_process_types:
            raise ValueError(f"Invalid process_type: {process_type}. Expected one of {valid_process_types}.")
        
        patches = self.collect_patch_paths(process_type=process_type)

        for patch_id, patch_paths in patches.items():
            # Load each patch file as an xarray DataArray
            patches_xarrays = [rioxarray.open_rasterio(path) for path in patch_paths]

            # Concatenate the DataArrays along a new 'time' dimension
            patch_ts = xr.concat(patches_xarrays, dim='time').to_dataset(name=var_name)

            # Define the folder path based on the patch type
            patches_folder = self.base_folder / "patches" / process_type
            patches_folder.mkdir(parents=True, exist_ok=True)

            # Construct the file path for the netCDF file
            patch_ts_filepath = patches_folder / f"{process_type}_{patch_id}.nc"  # e.g., .../IMG_patch_0_128.nc or .../MSK_patch_0_128.nc

            # Save the time series as a netCDF file
            patch_ts.to_netcdf(patch_ts_filepath, format="NETCDF4", engine="h5netcdf")
            patch_ts.close()
            print(f"----Successfully saved {patch_id} as netCDF")

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