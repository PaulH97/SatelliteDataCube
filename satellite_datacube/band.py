import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path 
from .utils import patchify

class SatelliteBand:
    """
    A class to represent a satellite band or mask. 

    Attributes:
        name (str): Name of the satellite band.
        path (Path): Path to the raster file of the satellite image.
        array (numpy.ndarray): Array representation of the band (shape: CxHxW).
        meta (dict): Metadata of raster file.
    """
    def __init__(self, band_name, band_path):
        """
        Constructs all the necessary attributes for the SatelliteBand object.
        Args:
            band_name (str): Name of the satellite band.
            band_path (str): File path to the raster data of the band.
        
        """
        self.name = band_name 
        self.path = band_path
        with rasterio.open(self.path, "r") as src:
            self.array = src.read() # shape(CxHxW)
            self.meta = src.meta.copy()

    def _calculate_scale_factor(self, new_resolution):
        """
        Calculate the scale factor for resampling based on new resolution.
        Args:
            new_resolution (int): The desired resolution in (m).
        Returns:
            float: The scale factor.
        """
        old_resolution = self.meta["transform"].a  # Assuming square pixels
        return old_resolution / new_resolution

    def _calculate_new_transform(self, scale_factor):
        """
        Calculate a new affine transform based on the scale factor.
        Args:
            scale_factor (float): The scale factor for resampling.
        Returns:
            Affine: The new affine transform.
        """
        old_transform = self.meta["transform"]
        return old_transform * old_transform.scale(1/scale_factor,1/scale_factor)

    def _calculate_new_dimensions(self, scale_factor):
        """
        Calculate new dimensions (width and height) of the raster based on the scale factor.
        Args:
            scale_factor (float): The scale factor for resampling.
        Returns:
            tuple: New dimensions (width, height) of the raster.
        """
        return int(self.meta["width"]*scale_factor), int(self.meta["height"]*scale_factor)
    
    def _update_metadata_to_new_resolution(self, new_resolution):
        """
        Update the metadata of the raster to reflect a new resolution.
        Args:
            new_resolution (float): The desired resolution.
        Returns:
            dict: Updated metadata.
        """
        scale_factor = self._calculate_scale_factor(new_resolution)
        new_transform = self._calculate_new_transform(scale_factor)
        new_width, new_height = self._calculate_new_dimensions(scale_factor)

        # Set up metadata for the new raster
        new_meta = self.meta
        new_meta.update({
            'transform': new_transform,
            'width': new_width,
            'height': new_height
        })
        return new_meta

    def _reproject_band(self, metadata):
        """
        Reproject the raster band based on new metadata.
        Args:
            metadata (dict): Metadata containing new projection information.
        Returns:
            Path: Path to the resampled raster file.
        """
        resampled_raster_path = self.path.parent / (self.path.stem + "_resampled.tif")

        with rasterio.open(self.path) as src:
            with rasterio.open(resampled_raster_path, 'w', **metadata) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=metadata["transform"],
                    dst_crs=metadata["crs"],
                    resampling=Resampling.bilinear
                )

        return resampled_raster_path
    
    def resample(self, new_resolution):
        """
        Resample the raster band to a new resolution.
        Args:
            new_resolution (float): The desired resolution.
        Returns:
            SatelliteBand: The current instance after resampling.
        """
        band_res = self.meta["transform"].a
        if band_res != new_resolution:
            print("Resampling")
            new_meta = self._update_metadata_to_new_resolution(new_resolution)
            resampled_band_path = self._reproject_band(new_meta)
            self.path = Path(resampled_band_path)
            with rasterio.open(self.path, "r") as src:
                self.array = src.read() 
                self.meta = src.meta.copy()
        return self

    def create_patches(self, patch_size):
        return patchify(source_array=self.array, patch_size=patch_size)

    def normalize_with_zscore(self):
        """
        Apply z-score normalization to the raster band array.
        Returns:
            numpy.ndarray: Normalized array.
        """
        return (self.array - self.array.mean()) / self.array.std()
         
    def normalize_with_minmax(self):
        """
        Apply min-max normalization to the raster band array.
        Returns:
            numpy.ndarray: Normalized array.
        """
        min_val, max_val = self.array.min(), self.array.max()
        min_boundary, max_boundary = 0, 1
        return (max_boundary - min_boundary) * ((self.array - min_val) / (max_val - min_val)) + min_boundary

    def stretch_contrast(self, percentiles=(4,92)):
        band_min = np.percentile(self.array, percentiles[0]).astype(np.float32)
        band_max = np.percentile(self.array, percentiles[1]).astype(np.float32)
        band_streched = (self.array - band_min) / (band_max - band_min)
        return band_streched

    def plot_histogram(self):
        """
        Plot histogram of the satellite band.
        """
        band_array = np.moveaxis(self.array, 0, -1) # HxWxC

        q25, q75 = np.percentile(band_array, [25, 75])
        bin_width = 2 * (q75 - q25) * len(band_array) ** (-1/3)
        bins = round((band_array.max() - band_array.min()) / bin_width)   
        bins = 300
         
        plt.figure(figsize=(10,10))
        plt.title("Band {} - Histogram".format(self.name), fontsize = 20)
        plt.hist(band_array.flatten(), bins = bins, color="lightcoral")
        plt.ylabel('Number of Pixels', fontsize = 16)
        plt.xlabel('DN', fontsize = 16)
        plt.show()
    
    def plot(self):
        """
        Plot satellite band data.
        """
        band_stretched = self.stretch_contrast()
        plt.figure(figsize=(10,10))
        plt.imshow(np.moveaxis(band_stretched, 0,-1), cmap="viridis")
        plt.title(f"{self.name}")
        plt.colorbar()
        plt.show()
