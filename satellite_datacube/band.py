import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pathlib import Path 

class SatelliteBand:
    def __init__(self, band_name, band_path):
        self.name = band_name 
        self.path = Path(band_path)
        with rasterio.open(self.path, "r") as src:
            self.array = src.read() # shape(CxHxW)
            self.meta = src.meta.copy()

    def _calculate_scale_factor(self, new_resolution):
        old_resolution = self.meta["transform"].a  # Assuming square pixels
        return old_resolution / new_resolution

    def _calculate_new_transform(self, scale_factor):
        old_transform = self.meta["transform"]
        return old_transform * old_transform.scale(1/scale_factor,1/scale_factor)

    def _calculate_new_dimensions(self, scale_factor):
        return int(self.meta["width"]*scale_factor), int(self.meta["height"]*scale_factor)
    
    def _update_metadata_to_new_resolution(self, new_resolution):
        # Calculate new transform and dimensions
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
        Resample a raster to a new resolution.
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

    def z_normalization(self):
        return (self.array - self.array.mean()) / self.array.std()
         
    def min_max_normalization(self):
        min_val, max_val = self.array.min(), self.array.max()
        min_boundary, max_boundary = 0, 1
        return (max_boundary - min_boundary) * ((self.array - min_val) / (max_val - min_val)) + min_boundary
           
    def plot_histo(self):
        
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
        plt.savefig("test_histo.png")
    
    def plot(self):
        
        plt.figure(figsize=(10,10))
        plt.imshow(np.moveaxis(self.array, 0,-1), cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.savefig("test_img.png")


    # def resample(self, resolution, reference_band_path, save_file=False):

    #     def reproject_band(source_dataset, source_band, destination, ref_transform):
    #         reproject(
    #             source=source_band,
    #             destination=destination,
    #             src_transform=source_dataset.transform,  # Use the transform from the dataset
    #             src_crs=source_dataset.crs,  # Use the CRS from the dataset
    #             dst_transform=ref_transform,
    #             dst_crs=source_dataset.crs,
    #             resampling=Resampling.bilinear
    #         )

    #     if self.band.res != (float(resolution), float(resolution)):
    #         with rasterio.open(reference_band_path) as ref_band:
    #             ref_transform = ref_band.transform
    #             ref_width = ref_band.width
    #             ref_height = ref_band.height

    #         meta = self.band.meta.copy()
    #         meta.update({
    #             'crs': self.band.crs,
    #             'transform': ref_transform,
    #             'width': ref_width,
    #             'height': ref_height
    #         })

    #         if save_file:
    #             output_path = os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + "_resampled.tif") 
    #             with rasterio.open(output_path, 'w', **meta) as dst:
    #                 reproject_band(self.band, rasterio.band(self.band, 1), rasterio.band(dst, 1), ref_transform)
    #             self.path = output_path 
    #             self.band = rasterio.open(self.path)
    #             self.band_arr = self.band.read()       
    #             return self

    #         else:
    #             resampled_array = np.empty((1, ref_height, ref_width))
    #             reproject_band(self.band, rasterio.band(self.band, 1), resampled_array[0,...], ref_transform)
    #             self.band_arr = resampled_array.astype(self.band_arr.dtype)
    #             return self

    #     else: 
    #         return self