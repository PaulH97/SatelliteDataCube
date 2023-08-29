import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
from .utils import patchify, select_patches, create_and_select_patches
from .image import Sentinel2
import random

# TODO:
# - exclude patches as self attribute -> we want to create multiple patches and not store a specific one in the data cube.. instead we want to use a load function to load it
# - same with dc parameters - we constantly want to chnage these values -> do not store them as attributes? Or create new datacube for new values? 
# - smarter if i init the datacube once and then use functions to change values -> 

# Should i use the class with a fixed timeseries and patches attribute? 
# So whenever i want to create patches with different sizes or different selected timesteps i need to initialize a new object -> exactly!!
# - i can implemnt self.patches, self.timeseries and self.global_mask

# when i want to chnage the patch_size or the selected ts the best way would be to create new instances! -> otherwise it is confusing - it is able but not recommended
# make it optinal to give the functions the parameters timeseries_length, patch_size or bad_pixel_limit

class SatelliteDataCube:
    def __init__(self, base_folder, parameters, load_data=True):

        self.base_folder = base_folder 
        self.timeseries_length = parameters["timeseries_length"]
        self.patch_size = parameters["patch_size"]
        self.bad_pixel_limit = parameters["bad_pixel_limit"]
        self.satellite_images = self.load_satellite_images()
        self.global_mask = None
        self.selected_timeseries = {} 
        self.patches = {}
        self.seed = 42

        if load_data:
            self.global_mask = self.load_global_mask()
            self.selected_timeseries = self.load_timeseries() # single selected ts that can be chnaged with function load_ts from json
            self.patches = self.load_patches()
            
        self._print_initialization_info()

    def _print_initialization_info(self) -> None:
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
        - base folder: /path/to/base_folder
        - length of timeseries: 12
        - limit of bad pixel per satellite image in timeseries: 5%
        - patch size of 256
        """
        divider = "-" * 20
        print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print("Initializing data-cube with following parameter:")
        print(f"- base folder: {self.base_folder}")
        print(f"- length of timeseries: {self.parameters['timeseriesLength']}")
        print(f"- limit of bad pixel per satellite image in timeseries: {self.parameters['badPixelLimit']}%")
        print(f"- patch size of {self.parameters['patchSize']}")

    def load_global_mask(self):
        print(f"Loading global_mask from: {self.base_folder}")
        global_mask_path = os.path.join(self.base_folder, "global_mask.npy")
        if os.path.exists(global_mask_path):
            return np.load(os.path.join(self.base_folder, "global_mask.npy"))
        else:
            print(f"Could not find global_mask file in: {self.base_folder}. If you want to create a global mask for the timeseries please use create_global_mask().")
            return None 

    def load_timeseries(self):
        print(f"Loading selected timeseries from: {self.base_folder}")
        selected_ts_path = os.path.join(self.base_folder, "selected_timeseries.json")
        if os.path.exists(selected_ts_path):
            with open(selected_ts_path, 'r') as file:
                timeseries = json.load(file)
                return timeseries[f"ts-{self.timseries_length}"]
        else:
            print(f"Could not find selected_timeseries file in: {self.base_folder}. If you want to create a global mask for the timeseries please use create_timeseries().")
            return None 

    def load_patches(self, patches_folder=""):
        """
        Load patches from the saved .npy files and format them as TCHW (Time, Channel, Height, Width).

        This function searches for saved patches in the base folder with names corresponding to 'img', 'msk', and 'msk_gb' and containing 
        information about the length of the timeseries. If found, it loads the patches into a dictionary, where the keys are the patch names 
        and the values are the loaded patches. If no patches are found, it returns an empty dictionary.

        Returns:
        - dict: A dictionary containing loaded patches. The potential keys are 'img', 'msk', and 'msk_gb', 
                and the values are the corresponding patches formatted as NTCHW.

        Example:
        If 'img_patches_ts-6.npy' exists in the patches folder and contains image patches, the returned dictionary might look like:
        {
            'img': np.ndarray of shape (num_patches, num_timesteps, num_channels, patch_height, patch_width),
            ...
        }
        """
        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        patches = {}
        for patchName in ["img", "msk", "msk_gb"]:
            patchPath = os.path.join(patches_folder, f"{patchName}_patches_ts-{self.timeseries_length}.npy")
            if os.path.exists(patchPath):
                patches[patchName] = np.load(patchPath)
        if patches:
            print("Loading patches as dictonary with keys [img, msk, msk_gb] from .npy files")
        else:
            print("Loading patches as empty dictonary")
        return patches

    def load_satellite_images(self):
        """
        Initialize satellite images by scanning the base folder and processing valid satellite image directories.

        This function scans the base folder for directories with numeric names, assuming each such directory 
        corresponds to a satellite image. For every valid directory found, an instance of the `Sentinel2` class 
        is created to represent the satellite image, and the resulting instances are stored in a dictionary 
        where the keys are the indices and the values are the `Sentinel2` instances.

        Note:
        The function assumes that valid satellite image directories in the base folder have numeric names.

        Returns:
        - dict: A dictionary where keys are indices of the satellite images and values are corresponding 
                `Sentinel2` instances representing the images.

        Example:
        Given a base folder structure like:
        base_folder/
        ├── 20180830/
        ├── 20180911/
        ├── temp/
        Calling `load_satellite_images()` will only process the '20180830' and '20180911' directories and return a dictionary 
        with their corresponding `Sentinel2` instances.
        """
        print("Initializing satellite images")
        return {
            i: Sentinel2(si_folder)
            for i, si_folder in enumerate(
                si_folder.path 
                for si_folder in os.scandir(self.base_folder) 
                if os.path.isdir(si_folder.path) and si_folder.name.isdigit()
            )
        }

    def load_or_built_global_mask(self):
        """
        Ensure that the global dataset is either loaded from existing storage or constructed afresh.

        This method checks if the global dataset is already present in the instance. If not, or if the dataset 
        doesn't contain information for the desired timeseries length specified in the instance's parameters, 
        it constructs the global data anew using the `create_global_data` method. If the global dataset is 
        present and meets the criteria, it simply loads it using the `load_global_data` method.

        Returns:
        - dict: A dictionary containing the global data. For detailed information see create_global_data()

        Example:
        If the global data for a specified timeseries length isn't loaded or doesn't meet the desired criteria, 
        the method might process and return global data containing elements like 'global_mask' and 'timeseries'.
        """
        if not self.global_mask:
            self.global_mask = self.create_global_mask()
        else:
            self.global_mask = self.load_global_mask()
        return 
    
    def load_or_built_timeseries(self):

        if not self.selected_timeseries:
            self.selected_timeseries = self.create_timeseries(timeseries_length=self.timeseries_length)
        else:
            self.selected_timeseries = self.load_timeseries()
        return 

    def load_or_built_patches(self, patches_folder=""):
        """
        Ensure that the satellite image patches are either loaded from existing storage or processed anew.

        This method first checks if patches are already loaded in the instance. If they aren't, or if their size 
        doesn't match the desired patch size specified in the instance's parameters, it processes the patches 
        using the `process_patches` method. If patches are already loaded and meet the size criteria, it attempts 
        to load them using the `load_patches_as_tchw` method.

        Returns:
        - dict: A dictionary containing the patches. The dictionary keys include 'img' for multi-spectral image patches,
                'msk' for binary mask patches corresponding to each timestep, and 'msk_gb' for a binary mask that 
                represents the entire timeseries without individual timesteps.

        Example:
        If previously processed patches aren't loaded or if they don't meet the desired size, the method might 
        process and return patches in a format like:
        {
            'img': np.ndarray of shape (num_patches, num_timesteps, num_channels, patch_height, patch_width),
            'msk': np.ndarray of shape (num_patches, num_timesteps, num_channels, patch_height, patch_width),
            'msk_gb': np.ndarray of shape (num_patches, num_channels, patch_height, patch_width)
        }
        """
        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        if os.path.exists(patches_folder):
            patches = self.load_patches(self.timeseries_length, patches_folder=patches_folder)
            if patches['img'].shape[-1] == self.patches_size:
                return patches
            else:
                patches = self.create_patches(patch_size=self.patches_size, timeseries_length=self.timeseries_length, output_folder=patches_folder)
        else:
            patches = self.create_patches(patch_size=self.patches_size, timeseriesLength=self.timeseries_length, output_folder=patches_folder)
        return patches

    def create_global_mask(self):
        """
        Generate a global mask for the data cube by aggregating masks from individual satellite images.

        This function iterates over all satellite images in the instance's 'satellite_images' attribute.
        For each image, the mask is initiated and checked. If any part of the mask has a value greater than or 
        equal to 1 (target class), it's considered a boolean mask (True for values >= 1). All such masks are aggregated to form 
        a global mask, which is stored in the instance's 'global_data' attribute under the key 'global_mask'.

        Note:
        The global mask is a logical OR aggregation of individual image masks, i.e., if a pixel is True in any 
        image mask, it will be True in the global mask.

        Example:
        If the satellite images have masks that highlight certain features or anomalies, 
        the global mask will highlight all these features across all images.
        """
        print("Building global mask of datacube")
        global_masks = []
        for satellite_image in self.satellite_images.values():
            satellite_image.initiate_mask()
            si_mask = satellite_image._mask
            if np.any(si_mask >= 1): 
                mask_bool = si_mask >= 1
                global_masks.append(mask_bool)
            satellite_image.unload_mask()
        self.global_mask = np.logical_or.reduce(global_masks).astype(int)
        return self.global_mask

    class SatelliteHandler:
    # Assuming other parts of the class are defined above this

    # def _get_image_by_index(self, idx):
    #     return list(self.satellite_images.values())[idx]

    # def _is_image_quality_acceptable(self, image):
    #     return image._badPixelRatio <= self.bad_pixel_limit and image not in self.timeseries

    # def _find_acceptable_neighbor(self, target_idx, max_search_limit=5):
    #     """Search for the nearest good quality image before and after the current index"""
    #     max_index = len(self.satellite_images.values()) - 1
    #     offset = 1
    #     while offset <= max_search_limit:
    #         for direction in [-1, 1]:
    #             new_idx = target_idx + (direction * offset)
    #             if 0 <= new_idx <= max_index:
    #                 neighbor = self._get_image_by_index(new_idx)
    #                 if self._is_image_quality_acceptable(neighbor):
    #                     return neighbor
    #         offset += 1
    #     return None

    # def _get_best_alternative(self, alternatives):
    #     alternatives.sort(key=lambda x: x[1])  # Sort by bad pixel ratio (best first)
    #     for alternative in alternatives:
    #         si = alternative[0]
    #         if self._is_image_quality_acceptable(si):
    #             return si
    #     return None

    # def create_timeseries(self):
    #     print(f"Selecting timeseries of length {self.timeseries_length} with bad pixel limit of {self.bad_pixel_limit} % for each satellite image")

    #     self.timeseries = []
    #     max_index = len(self.satellite_images.values()) - 1
    #     selected_indices = np.linspace(0, max_index, self.timeseries_length, dtype=int)

    #     for target_idx in selected_indices:
    #         print("[" + " ".join(str(x) for x in range(len(self.timeseries) + 1)) + "]", end='\r')
            
    #         satellite_image = self._get_image_by_index(target_idx)
    #         if self._is_image_quality_acceptable(satellite_image):
    #             self.timeseries.append(satellite_image)
    #         else:
    #             acceptable_neighbor = self._find_acceptable_neighbor(target_idx)
    #             if acceptable_neighbor:
    #                 self.timeseries.append(acceptable_neighbor)
    #             else:
    #                 alternatives = [(img, img._badPixelRatio) for idx, img in self.satellite_images.items() if img not in self.timeseries]
    #                 best_alternative = self._get_best_alternative(alternatives)
    #                 if best_alternative:
    #                     self.timeseries.append(best_alternative)

    #     self.timeseries = sorted(self.timeseries, key=lambda image: image.date)
    #     tsIdx = [idx for idx, si in self.satellite_images.items() if si in self.timeseries]
    #     tsDate = [int(si.date.strftime("%Y%m%d")) for si in self.timeseries]

    #     self.selected_timeseries[f"ts-{self.timeseries_length}"] = np.array([tsIdx, tsDate])
    #     return self.selected_timeseries

    
    
    def create_timeseries(self):
        """
        Select a timeseries of satellite images based on specified length and bad pixel limit.

        The function aims to select a set of satellite images that fall within the defined 
        timeseries length while ensuring that each image's bad pixel ratio does not exceed 
        the specified limit. If a target image exceeds the bad pixel ratio, the function 
        searches for neighboring images that meet the criteria. 

        Parameters:
        - timeseries_length (int): Desired number of satellite images in the timeseries.
        - bad_pixel_limit (float): Maximum acceptable bad pixel ratio (in percentage) for each image.

        Returns:
        - np.ndarray: A 2D array where the first row contains indices of selected satellite images 
                        and the second row contains corresponding dates in YYYYMMDD format.

        Note:
        The function also updates the instance's global_data attribute with the selected timeseries data.

        Example:
        Given satellite_images = {0: SatelliteImage(date="20220101"), 1: SatelliteImage(date="20220201"), ...}
        If selected images are from indices [0, 5, 10], with dates ["20220101", "20220601", "20221101"],
        the return value will be:
        array([[     0,      5,     10],
                [20220101, 20220601, 20221101]])
        """        
        print(f"Selecting timeseries of length {self.timeseries_length} with bad pixel limit of {self.bad_pixel_limit} % for each satellite image")
        timeseries = []
        max_index = len(self.satellite_images.values()) - 1
        selected_indices = np.linspace(0, max_index, self.timeseries_length, dtype=int)
        for target_idx in selected_indices:
            print("[" + " ".join(str(x) for x in range(len(timeseries) + 1)) + "]", end='\r')
            satellite_image = list(self.satellite_images.values())[target_idx]
            if satellite_image._badPixelRatio <= self.bad_pixel_limit and satellite_image not in timeseries:
                timeseries.append(satellite_image)
            else:
                # Search for the nearest good quality image before and after the current index
                offset = 1
                found_good_image = False
                max_search_limit = 5
                alternatives = []
                while not found_good_image and offset <= max_search_limit:
                    # Try looking both before and after
                    for direction in [-1, 1]:
                        # Calculate new index
                        new_idx = target_idx + (direction * offset)
                        # Check if the new index is valid
                        if 0 <= new_idx <= max_index:
                            neighbor_satellite_image = list(self.satellite_images.values())[new_idx]
                            # If the neighboring image is useful, append it
                            if satellite_image._badPixelRatio <= self.bad_pixel_limit and satellite_image not in timeseries:
                                timeseries.append(neighbor_satellite_image)
                                found_good_image = True
                                break  # Exit the inner loop
                            else:
                                # Add to alternative list with its bad pixel ratio
                                alternatives.append((neighbor_satellite_image, neighbor_satellite_image._badPixelRatio))
                    # Increase the offset to look further
                    offset += 1
                # If limit reached, add the image with the lowest bad pixel ratio that is not in timeseries already
                if not found_good_image:
                    alternatives.sort(key=lambda x: x[1])  # Sort by bad pixel ratio (best first)
                    for alternative in alternatives:
                        si = alternative[0]
                        if si not in timeseries:
                            timeseries.append(si)
                            break
                        else:
                            continue
        timeseries = sorted(timeseries, key=lambda image: image.date)
        tsIdx = [idx for idx, si in self.satellite_images.items() if si in timeseries]
        tsDate = [int(si.date.strftime("%Y%m%d")) for si in self.satellite_images.values() if si in timeseries]
        self.selected_timeseries[f"ts-{self.timeseries_length}"] = np.array([tsIdx, tsDate])
        return self.selected_timeseries
         
    def create_patches(self, source, indices=False):
        if not self.selected_timeseries:
            self.load_or_built_timeseries()
        si_timeseries = [self.satellite_images[idx] for idx in self.selected_timeseries[0]] # [1] for getting date of ts 
        patches = {source: []} # source could be img, msk
        for image in si_timeseries:
            patches = image.process_patches(self.patch_size, source=source, indices=indices) # returns list of patches
            patches[source].append(patches)
            image.unload_bands()
            image.unload_mask()
        # convert it to an array of pattern NxTxCxHxW where N is the number of satellite images in timeseries
        self.patches[source] = np.swapaxes(np.array(patches),0,1) 
        return self.patches

    def select_patches(self, class_values, seed, class_ratio=(100,0)):
        if not self.global_mask:
            self.load_or_built_global_mask()
        random.seed(seed)
        class_ratio= [i / 100 for i in class_ratio]
        global_mask_patches = patchify(self.global_mask, self.patch_size)
        class_indices = [idx for idx, patch in enumerate(global_mask_patches) if np.any(np.isin(patch, class_values))]
        no_class_indices = [idx for idx, patch in enumerate(global_mask_patches) if not np.any(np.isin(patch, class_values))]
        num_noclass_patches = int((len(class_indices) / class_ratio[0]) * class_ratio[1])
        no_class_indices = random.sample(no_class_indices, num_noclass_patches)
        return class_indices + no_class_indices
            
    def filter_patches(self, patches_idx):
        for patchType, patchArray in self.patches.items():
            patches = [patchArray[idx] for idx in patches_idx]
            self.patches[patchType] = patches    
        return self.patches
        
    def save_patches(self, patches_folder=""):
        patches_folder = os.path.join(self.base_folder, "patches") if not patches_folder else patches_folder
        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        for patchType, patchArray in self.patches.items():
            print(patchType, patchArray.shape)  
            np.save(os.path.join(patches_folder, f"{patchType}_patches_ts-{self.timeseries_length}.npy"), patchArray)
        return
    
    def process_patches(self, sources, indices=False, patches_folder=""):
        
        patches_idx = self.select_patches(class_values=[1])
        for source in sources:
            if source in ["img", "msk"]:
                self.patches[source] = self.create_patches(source=source, indices=indices, patches_folder=patches_folder)
            else:
                self.patches[source] = patchify(self.global_mask, self.patch_size)
        self.filter_patches(patches_idx=patches_idx)
        self.save_patches()
        return self.patches
           	
    def process_patches(self, patch_size, patch_folder):
        """
        Preprocess the satellite data to prepare it for further analysis or modeling.

        The preprocessing steps include:
        1. Building or loading global data: This ensures that the necessary global datasets, 
        which might include various datasets representing different aspects or metadata 
        of the satellite images, are available. If required data for the desired timeseries 
        length isn't already present, it is built using the `create_global_data` method. 
        Otherwise, it's loaded using the `load_global_data` method.
        
        2. Creating or loading patches: This step divides satellite images into patches for 
        both image and mask data. If patches meeting the desired size aren't already loaded, 
        they are processed using the `process_patches` method. If they are present, they are 
        loaded using the `load_patches_as_tchw` method.

        Attributes Updated:
        - global_data: Contains the global datasets after the preprocessing.
        - patches: Contains the processed patches after the preprocessing.
        
        Returns:
        None
        """
        self.load_or_built_global_mask()
        self.load_or_built_timeseries()
        
        return patches

    def sanity_check(self, patches):
        return

    def create_spectral_signature(self, shapefile, save_csv=True):
        
        from rasterio.mask import mask
        import pandas as pd 

        geometries = []
        # Open your shapefile
        #file = ogr.Open(shapefile)
        layer = file.GetLayer(0)
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            geometry = json.loads(feature.GetGeometryRef().ExportToJson())
            geometries.append(geometry)
        
        file = None
        spectral_sig = {}
        
        for satellite_image in self.satellite_images.values():
           
            satellite_image.initiate_bands()
        
            for b_name, band in satellite_image._bands.items():
                                
                band_value = 0
                
                with rasterio.open(band.path) as src:
                    
                    # Loop over polygons and extract raster values
                    for polygon in geometries:
                        
                        out_image, out_transform = mask(src, [polygon], crop=True)                         
                        band_value += np.mean(out_image)

                mean_value = band_value/len(geometries)
                
                if b_name not in spectral_sig:

                    spectral_sig[b_name] = [mean_value] # {B02:[1,2,3,4,5...90], B03:[1,2,3,4,5...90],...}
                else:
                    spectral_sig[b_name].append(mean_value)

            satellite_image.unload_bands()

        spectral_sig["Timestamp"] = [satellite_image.date for satellite_image in self.satellite_images.values()]
        spectral_sig_df = pd.DataFrame(spectral_sig)
        spectral_sig_df = spectral_sig_df.set_index('Timestamp').reset_index()

        spectral_sig_df.to_csv("spectral_signature.csv")

        return spectral_sig_df  
    
    def plot_band_signature(self, spectral_signature):
        # spectral_signature["NDVI"] = (spectral_signature["B08"] - spectral_signature["B04"])/(spectral_signature["B08"] + spectral_signature["B04"])
        # plt.plot(spectral_signature['Timestamp'], spectral_signature["NDVI"], label="NDVI", marker='o')
        plt.figure(figsize=(10, 8))
        for column in spectral_signature.columns[1:]:  # Assuming first column is 'Timestamp'"B02_DWN_log_[10, 90]_histo.png"
            plt.plot(spectral_signature['Timestamp'], spectral_signature[column], label=column)
        plt.xlabel('Year')
        plt.ylabel('Change')
        plt.title('Annual Change of Different Bands')
        plt.legend()
        plt.show()
        return
