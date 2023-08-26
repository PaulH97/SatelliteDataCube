import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
import json

class SatelliteBand:
    def __init__(self, band_name, band_path):
        self.name = band_name 
        self.path = band_path
        self.band = rasterio.open(self.path)
        self.bandArray = self.band.read() # shape(CxHxW)

    def get_metadata(self):
        metadata = self.band.meta.copy()
        metadata['name'] = self.name
        metadata['path'] = self.path
        
        return metadata

    def export_metadata_to_json(self, filename):
        with open(filename, 'w') as json_file:
            json.dump(self.get_metadata(), json_file, indent=4)

    def get_raster_properties(self):
        
        crs = self.band.crs
        resolution = self.band.res
        bounds = self.band.bounds
        num_bands = self.band.count
        dtype = self.band.dtypes[0]
        width = self.band.width
        height = self.band.height
        transform = self.band.transform

        return {
            'crs': crs,
            'resolution': resolution,
            'bounds': bounds,
            'num_bands': num_bands,
            'dtype': dtype,
            'width': width,
            'height': height,
            'transform': transform
        }
    
    def resample(self, resolution, reference_band_path, save_file=False):

        def reproject_band(source_dataset, source_band, destination, ref_transform):
            from rasterio.enums import Resampling
            from rasterio.warp import reproject

            reproject(
                source=source_band,
                destination=destination,
                src_transform=source_dataset.transform,  # Use the transform from the dataset
                src_crs=source_dataset.crs,  # Use the CRS from the dataset
                dst_transform=ref_transform,
                dst_crs=source_dataset.crs,
                resampling=Resampling.bilinear
            )

        if self.band.res != (float(resolution), float(resolution)):
            with rasterio.open(reference_band_path) as ref_band:
                ref_transform = ref_band.transform
                ref_width = ref_band.width
                ref_height = ref_band.height

            meta = self.band.meta.copy()
            meta.update({
                'crs': self.band.crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height
            })

            if save_file:
                output_path = os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + "_resampled.tif") 
                with rasterio.open(output_path, 'w', **meta) as dst:
                    reproject_band(self.band, rasterio.band(self.band, 1), rasterio.band(dst, 1), ref_transform)
                self.path = output_path       
                return self

            resampled_array = np.empty((1, ref_height, ref_width))
            reproject_band(self.band, rasterio.band(self.band, 1), resampled_array[0,...], ref_transform)
            return resampled_array.astype(self.bandArray.dtype)

        else: 
            return self.bandArray

    def normalize(self, method="z-score"):
                
        if method == "min-max":
            min_val, max_val = self.bandArray.min(), self.bandArray.max()
            min_boundary, max_boundary = 0, 1
            norm_data = (max_boundary - min_boundary) * ((self.bandArray - min_val) / (max_val - min_val)) + min_boundary
            
        elif method == "z-score":
            mean = self.bandArray.mean()
            std = self.bandArray.std()
            norm_data = (self.bandArray - mean) / std

        else:
            raise ValueError("Invalid normalization method. Choose 'min-max' or 'z-score'.")
            
        return norm_data
    
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
    
    def plot_band(self):
        
        plt.figure(figsize=(10,10))
        plt.imshow(np.moveaxis(self.array, 0,-1), cmap="viridis")
        plt.colorbar()
        plt.show()
