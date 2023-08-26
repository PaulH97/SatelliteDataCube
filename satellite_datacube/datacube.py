import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
import h5py
from utils import patchify, select_patches
from image import Sentinel2

class SatelliteDataCube:
    def __init__(self, base_folder, parameters, preprocess=True):
        print(f"-------------------- {os.path.basename(base_folder)} --------------------")
        print("Initializing data-cube with following parameter:")
        print(f"- base folder: {base_folder}")
        print(f"- length of timeseries: {parameters['timeseriesLength']}") 
        print(f"- limit of bad pixel per satellite image in timeseries: {parameters['badPixelLimit']}%")
        print(f"- patch size of {parameters['patchSize']}")
        self.base_folder = base_folder # D:\WeMonitor\Data\Landslides\S2\Thrissur
        self.satellite_images = self.initiate_satellite_images()
        self.global_data_file = os.path.join(self.base_folder, "global_data.hdf5")
        self.global_data = self.load_global_data() if os.path.exists(self.global_data_file) else {}
        self.patches = self.loadPatchesAsTCHW()
        self.seed = 42
        if preprocess:
            if not self.global_data or f"ts{parameters['timeseriesLength']}" not in self.global_data:
                self.global_data = self.build_global_data(parameters["timeseriesLength"], parameters["badPixelLimit"])
            if not self.patches or self.patches[next(iter(self.patches))].shape[-1] != parameters["patchSize"]:
                self.patches = self.process_patches(parameters["patchSize"], parameters["timeseriesLength"])
           
    def initiate_satellite_images(self):
        print("Initializing satellite images")
        return {
            i: Sentinel2(si_folder)
            for i, si_folder in enumerate(
                si_folder.path 
                for si_folder in os.scandir(self.base_folder) 
                if os.path.isdir(si_folder.path) and si_folder.name.isdigit()
            )
        }

    def save_global_data(self):
        print(f"Saving global data in: {self.global_data_file}")
        with h5py.File(self.global_data_file, 'a') as hf: 
            for key, value in self.global_data.items():
                if key in hf:
                    del hf[key] # delete the existing dataset so it can be overwritten
                hf.create_dataset(key, data=value)

    def load_global_data(self):
        print(f"Loading global data from: {self.global_data_file}")
        data_dict = {}
        with h5py.File(self.global_data_file, 'r') as hf:
            for key in hf.keys():
                data_dict[key] = np.array(hf[key])
        return data_dict

    def loadPatchesAsTCHW(self):
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

    def build_global_data(self, timeseries_length, bad_pixel_limit):
        global_data_methods = {'global_mask': self.build_global_mask,'timeseries': lambda: self.select_timeseries(timeseries_length, bad_pixel_limit)}
        for method in global_data_methods.values():    
                method()
        self.save_global_data()
        return self.global_data

    def build_global_mask(self):
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
    
    def select_timeseries(self, timeseries_length, bad_pixel_limit ):
        
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
        Create useful patches in the datacube folder with shapes of : T x C x H x W 
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

    def spectral_signature(self, shapefile, save_csv=True):
        
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
    
    def plot_signatures(self, spectral_signature):
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
