import os
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
from .utils import patchify, select_equally_spaced_items
from .image import Sentinel1, Sentinel2
from .annotation import Sentinel2Annotation, Sentinel1Annotation
import random
import pandas as pd
from datetime import datetime
from glob import glob
from pathlib import Path

# TODO: Update the function with storing the shapefile in a varibale -> creating mask with that and not using the annotation.tif?
# maybe i should select timeseries and store it not as instance variable?
class SatelliteDataCube:
    def __init__(self):
        self.base_folder = None
        self.satellite = None
        self.images_by_date = {}
        self.selected_images_by_date = {}
        self.annotation = None
        self.spectral_signature = None

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
    
    def select_images_with_dates(self, dates):
        self.selected_images_by_date = {date: self.images_by_date[date] for date in dates if date in self.images_by_date}
        return self.selected_images_by_date

    def create_spectral_signature(self, annotation_shapefile):
        datacube_spectral_sig = {}
        for image_date, image in self.images_by_date.items():
            image_spectral_sig = image.calculate_spectral_signature(annotation_shapefile)
            datacube_spectral_sig[image_date] = image_spectral_sig
        return datacube_spectral_sig
    
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
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images_by_date = self._load_satellite_images()
        self.annotation = self._load_annotation()
        self._print_initialization_info()
     
    def _load_annotation(self):
        annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
        s2_satellite_image = next(iter(self.images_by_date.values()))
        return Sentinel2Annotation(s2_satellite_image, annotation_shapefile)
    
    def _load_satellite_images(self):
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                images_by_date[date_satellite_image] = Sentinel2(satellite_image_folder, date_satellite_image)
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
            raise ValueError(f"An error occurred while updating the satellite images of data-cube:{ValueError}. 
                             Please make sure that you first select images with the functions select_images_by date() or select_imgaes_by_date().")
        return 
            
class Sentinel1DataCube(SatelliteDataCube):
    def __init__(self, base_folder):
        super().__init__()
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.satellite_images_folder = self.base_folder / self.satellite
        self.images_by_date = self._load_satellite_images()
        self.annotation = self._load_annotation()
        self._print_initialization_info()
     
    def _load_annotation(self):
        annotation_shapefile = [file for folder in self.base_folder.iterdir() if folder.name == 'annotations' for file in folder.glob("*.shp")][0]
        s2_satellite_image = next(iter(self.images_by_date.values()))
        return Sentinel2Annotation(s2_satellite_image, annotation_shapefile)
    
    def _load_satellite_images(self):
        images_by_date = {}
        for satellite_image_folder in self.satellite_images_folder.iterdir():
            if satellite_image_folder.is_dir():
                date_satellite_image = datetime.strptime(satellite_image_folder.name, "%Y%m%d").date()
                images_by_date[date_satellite_image] = Sentinel1(satellite_image_folder, date_satellite_image)
                satellite_images_by_date_sorted = dict(sorted(images_by_date.items()))
                return satellite_images_by_date_sorted

    def _find_higher_quality_satellite_image(self, satellite_image, search_limit=5):
        pass

    def find_best_selected_images(self):
        pass
           