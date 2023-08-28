import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
import h5py
from .utils import patchify, select_patches
from .image import Sentinel2

class SatelliteDataCube:
    def __init__(self, base_folder, parameters, load_data=True):
        """
        Initialize an instance of the SatelliteDataCube class.

        This constructor sets up the necessary attributes for the SatelliteDataCube based on the provided parameters. 
        It also has an option to immediately load data upon initialization, which can be controlled with the `load_data` flag.

        Parameters:
        - base_folder (str): Path to the base directory that contains the satellite data. This directory might include 
                            satellite images, global data, patches, and other related data. The satellite images should be stored 
                            in this base_folder as directories with the date as name like 20180830.
        - parameters (dict): A dictionary containing various configuration parameters for preprocessing and other operations. 
                            Expected keys include 'timeseriesLength', 'badPixelLimit', and 'patchSize'.
        - load_data (bool, optional): A flag to determine whether the satellite data should be loaded immediately upon 
                                    initialization. If set to True, the constructor will attempt to load satellite images, 
                                    global data, and patches. Defaults to True.

        Attributes:
        - parameters (dict): Stores the provided preprocessing parameters.
        - base_folder (str): Holds the path to the base directory containing satellite data.
        - global_data_file (str): Path to the HDF5 file that might contain global data.
        - seed (int): Seed value for any random operations to ensure reproducibility.
        - satellite_images (dict, optional): Dictionary of satellite images, initialized only if `load_data` is True.
        - global_data (dict, optional): Dictionary containing global data (selected timeseries + global mask), loaded or initialized based on the presence 
                                        of the global data file and the `load_data` flag.
        - patches (dict, optional): Dictionary containing processed patches, loaded if `load_data` is True.

        Example:
        To initialize a SatelliteDataCube with specific parameters and immediate data loading:
        ```
        parameters = {
            'timeseriesLength': 12,
            'badPixelLimit': 5,
            'patchSize': 256
        }
        cube = SatelliteDataCube("/path/to/base_folder", parameters)
        ```
        """
        self.parameters = parameters
        self.base_folder = base_folder
        self.global_data_file = os.path.join(self.base_folder, "global_data.hdf5")
        self.seed = 42

        self._print_initialization_info()

        if load_data:
            self.satellite_images = self.initiate_satellite_images()
            self.global_data = self.load_global_data() if os.path.exists(self.global_data_file) else {}
            self.patches = self.load_patches_as_tchw()

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

    def preprocess(self):
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
        self.global_data = self._ensure_global_data_loaded_or_built()
        self.patches = self._ensure_patches_loaded_or_processed()
        return

    def _ensure_global_data_loaded_or_built(self):
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
        if not self.global_data or f"ts{self.parameters['timeseriesLength']}" not in self.global_data:
            return self.create_global_data(self.parameters["timeseriesLength"], self.parameters["badPixelLimit"])
        return self.load_global_data()

    def _ensure_patches_loaded_or_processed(self):
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
        if not self.patches or self.patches[next(iter(self.patches))].shape[-1] != self.parameters["patchSize"]:
            return self.process_patches(self.parameters["patchSize"], self.parameters["timeseriesLength"])
        return self.load_patches_as_tchw()

    def initiate_satellite_images(self):
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
        Calling `initiate_satellite_images()` will only process the '20180830' and '20180911' directories and return a dictionary 
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
    
    def create_global_data(self, timeseries_length, bad_pixel_limit):
        """
        Construct and store a global dataset based on specified timeseries length and a bad pixel threshold.

        This function creates a global dataset that includes:
        1. A global mask which aggregates mask information from all satellite images.
        2. A selected timeseries of satellite images that adhere to the provided timeseries length and 
        bad pixel limit criteria.

        After constructing the global dataset, it's saved using the `save_global_data` method to ensure persistence.

        Parameters:
        - timeseries_length (int): Desired number of satellite images in the timeseries.
        - bad_pixel_limit (float): Maximum allowable percentage of bad pixels in an image for it to be 
                                considered in the timeseries.

        Returns:
        - dict: A dictionary containing the constructed global data. The keys include 'global_mask' for the 
                aggregated mask and 'timeseries' for the selected sequence of satellite images.

        Example:
        Given satellite images with varying levels of bad pixels, calling `create_global_data(10, 5)` 
        might return a dictionary with a global mask and a timeseries of 10 images, each having 
        less than 5% bad pixels.
        """
        global_data_methods = {'global_mask': self.create_global_mask,'timeseries': lambda: self.select_timeseries(timeseries_length, bad_pixel_limit)}
        for method in global_data_methods.values():    
                method()
        self.save_global_data()
        return self.global_data

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
        self.global_data["global_mask"] = np.logical_or.reduce(global_masks).astype(int)
        return 

    def load_global_data(self):
        """
        Retrieve the global data of satellite images from an HDF5 file stored in the base folder.

        This function reads from the HDF5 file specified by the instance's 'global_data_file' attribute, 
        which may contain various datasets representing different aspects or metadata of satellite images. 
        The datasets are loaded into a dictionary and returned.

        Note:
        The loaded file's path is determined by the instance's 'global_data_file' attribute.

        Returns:
        - dict: A dictionary where the keys are the dataset names and the values are the corresponding loaded data arrays.

        Example:
        If the HDF5 file contains datasets named 'timeseries' and 'metadata', the returned dictionary might look like:
        {
            'timeseries': np.ndarray,
            'global_mask': np.ndarray,
            ...
        }
        """
        print(f"Loading global data from: {self.global_data_file}")
        data_dict = {}
        with h5py.File(self.global_data_file, 'r') as hf:
            for key in hf.keys():
                data_dict[key] = np.array(hf[key])
        return data_dict

    def save_global_data(self):
        """
        Save the global data of the satellite images to an HDF5 file.

        This function saves the global data to an HDF5 file located in the base folder. It contains the selected steps (dates) from the timeseries and a global mask for
        the entire timeseries of the data cube. If any datasets with the same key already exist in the HDF5 file,
        they will be deleted and replaced with the new data.

        Note:
        The saved file's path is determined by the instance's 'global_data_file' attribute.

        Example:
        If the instance's global_data contains:
        {
            'timeseries': np.ndarray,
            'global_mask': np.ndarray,
            ...
        }
        Calling `save_global_data()` will save these arrays to the specified HDF5 file in the base folder.
        """
        print(f"Saving global data in: {self.global_data_file}")
        with h5py.File(self.global_data_file, 'a') as hf: 
            for key, value in self.global_data.items():
                if key in hf:
                    del hf[key] # delete the existing dataset so it can be overwritten
                hf.create_dataset(key, data=value)

    def select_timeseries(self, timeseries_length, bad_pixel_limit ):
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
    
        def is_useful_image(satellite_image, bad_pixel_limit, timeseries):
            satellite_image.calculate_bad_pixels()
            satellite_image.unload_mask()
            return satellite_image._badPixelRatio <= bad_pixel_limit and satellite_image not in timeseries
        
        print(f"Selecting timeseries of length {timeseries_length} with bad pixel limit of {bad_pixel_limit} % for each satellite image")
        timeseries = []
        max_index = len(self.satellite_images.values()) - 1
        selected_indices = np.linspace(0, max_index, timeseries_length, dtype=int)
        
        for target_idx in selected_indices:
            print("[" + " ".join(str(x) for x in range(len(timeseries) + 1)) + "]", end='\r')
            satellite_image = list(self.satellite_images.values())[target_idx]
            if is_useful_image(satellite_image, bad_pixel_limit, timeseries):
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
                            if is_useful_image(neighbor_satellite_image, bad_pixel_limit, timeseries):
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
        self.global_data[f"ts{timeseries_length}"] = np.array([tsIdx, tsDate])
        return np.array([tsIdx, tsDate])
 
    def process_patches(self, patch_size, timeseriesLength, ratio_classes=(100,0), indices=False):
        """
        Process and create patches from satellite images based on the specified patch size and timeseries length.

        This function divides satellite images into patches and optionally selects a subset of them based on 
        a given ratio between target classes and the background. The patches are stored in a dictionary with 
        keys representing different types of data and values being the patches themselves, following the TxCxHxW format:
        - T: Time
        - C: Channels (bands)
        - H: Height
        - W: Width

        The dictionary contains:
        - 'img': Patches from multi-spectral images for each timestep.
        - 'msk': Binary mask patches corresponding to each timestep.
        - 'msk_gb': A single binary mask for the entire timeseries, without the time axis.

        Parameters:
        - patch_size (int): Desired size of the patches.
        - timeseries_length (int): Length of the timeseries, used to retrieve the appropriate timeseries from global data.
        - ratio_classes (tuple, optional): Tuple containing the ratio between target class and background. Defaults to (100,0).
        - indices (bool, optional): If true, specific indices will be used. Defaults to False.

        Returns:
        - dict: A dictionary containing processed patches. The keys are 'img', 'msk', and 'msk_gb', 
                and the values are the corresponding patches.

        Example:
        Given a timeseries of satellite images, calling `process_patches(256, 5)` 
        might return a dictionary where:
        'img': (74,5,10,256,256) 
        'msk': (74,5,1,256,256)
        'msk_gb': (74,1,256,256)
        """
        print(f"Creating patches with size {patch_size} and ratio of {ratio_classes} between target class and background")
        def create_and_select_patches(image, selected_indices):
            X_patches = image.process_patches(patch_size, source="img", indices=indices)
            y_patches = image.process_patches(patch_size, source="msk", indices=indices)
            X_selected_patches = [X_patches[idx] for idx in selected_indices]
            y_selected_patches = [y_patches[idx] for idx in selected_indices]
            image.unload_bands()
            image.unload_mask()
            return X_selected_patches, y_selected_patches
        
        global_mask_patches = patchify(self.global_data["global_mask"], patch_size)
        selected_indices = select_patches(global_mask_patches, ratio_classes, seed=self.seed)
        si_timeseries = [self.satellite_images[idx] for idx in self.global_data[f"ts{timeseriesLength}"][0]] # [1] for getting date of ts 
        patches = {"img": [], "msk": []}
        for image in si_timeseries:
            print(f"Start with satellite image at {image.date} in timeseries")
            X_selected_patches, y_selected_patches = create_and_select_patches(image, selected_indices)
            patches["img"].append(X_selected_patches)
            patches["msk"].append(y_selected_patches)
        
        patches = {patchType: np.swapaxes(np.array(patchValues),0,1) for patchType, patchValues in patches.items()}
        patches["msk_gb"] = np.array([global_mask_patches[idx] for idx in selected_indices])
        for patchType, patchArray in patches.items():
            print(patchType, patchArray.shape)  
            np.save(os.path.join(self.base_folder, f"{patchType}_patches.npy"), patchArray)
        return patches

    def load_patches_as_tchw(self):
        """
        Load patches from the saved .npy files and format them as TCHW (Time, Channel, Height, Width).

        This function searches for saved patches with names corresponding to 'img', 'msk', and 'msk_gb' 
        in the base folder. If found, it loads the patches into a dictionary, where the keys are the patch names 
        and the values are the loaded patches. If no patches are found, it returns an empty dictionary.

        Returns:
        - dict: A dictionary containing loaded patches. The potential keys are 'img', 'msk', and 'msk_gb', 
                and the values are the corresponding patches formatted as NTCHW.

        Example:
        If 'img_patches.npy' exists in the base folder and contains image patches, the returned dictionary might look like:
        {
            'img': np.ndarray of shape (num_patches, num_timesteps, num_channels, patch_height, patch_width),
            ...
        }
        """
        patches = {}
        for patchName in ["img", "msk", "msk_gb"]:
            patchPath = os.path.join(self.base_folder, f"{patchName}_patches.npy")
            if os.path.exists(patchPath):
                patches[patchName] = np.load(patchPath)
        if patches:
            print("Loading patches as dictonary with keys [img, msk, msk_gb] from .npy files")
        else:
            print("Loading patches as empty dictonary")
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
