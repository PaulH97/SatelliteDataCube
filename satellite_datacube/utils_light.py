import re
from rasterio.enums import Resampling
import os
from pathlib import Path
import pandas as pd
from affine import Affine
import numpy as np 
import xarray as xr

def resample_raster(raster, target_resolution=10):
    current_resolution = raster.rio.resolution()
    scale_factor = current_resolution[0] / target_resolution
    new_shape = (int(raster.rio.shape[0] * scale_factor), int(raster.rio.shape[1] * scale_factor))
    resampled_raster = raster.rio.reproject(raster.rio.crs,shape=new_shape,resampling=Resampling.bilinear)
    return resampled_raster

def extract_S2_band_name(file_name):
    pattern = r"(B\d+[A-Z]?|SCL)\.tif"
    match = re.search(pattern, str(file_name))
    return match.group(1) if match else None

def extract_S1_band_name(file_name):
    pattern = r'(vv|vh).*\.tif'
    match = re.search(pattern, str(file_name))
    return match.group(1) if match else None

def extract_band_number(key):
    order = {"SCL": 100}
    return order.get(key, int(re.findall(r'\d+', key)[0]) if re.findall(r'\d+', key) else float('inf'))

def available_workers(reduce_by=1):
    """Calculate the number of available workers for Dask."""
    total_cores = os.cpu_count()
    load_average = os.getloadavg()[0]  # Get 1-minute load average
    free_cores = max(1, min(total_cores, int(total_cores - load_average)))
    return max(1, free_cores - reduce_by)

def normalize(array):
    # Normalize the bands for plotting
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))

def S2_xds_preprocess(ds):
    # Extract the date from the file path and set as a time coordinate
    date_str = Path(ds.encoding['source']).parent.stem
    ds = ds.expand_dims(time=[pd.to_datetime(date_str)])

    band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
    for i, band_name in enumerate(band_names):
        ds[band_name] = ds['band_data'].isel(band=i)
    
    # Drop the original multi-band data variable if it is no longer needed
    ds = ds.drop_vars('band_data')
    return ds

def S1_xds_preprocess(ds):
    # Extract the date from the file path and set as a time coordinate
    date_str = Path(ds.encoding['source']).parent.stem
    ds = ds.expand_dims(time=[pd.to_datetime(date_str)])

    band_names = ["VH", "VV"]
    for i, band_name in enumerate(band_names):
        ds[band_name] = ds['band_data'].isel(band=i)
    
    ds = ds.drop_vars('band_data')
    return ds

def extract_transform(xds):
    new_order_indices = [1, 2, 0, 4, 5, 3] 
    original_transform = xds.spatial_ref.attrs['GeoTransform'].split(" ")
    original_transform = [float(val) for val in original_transform]
    original_transform = [original_transform[i] for i in new_order_indices]
    return Affine(*original_transform)

def fill_nans(xds, fill_value=-9999):
    for var_name in xds.variables:
        if var_name not in xds.coords and var_name not in xds.attrs:          
            xds[var_name].encoding['_FillValue'] = fill_value
    return xds

def pad_patch(patch, patch_size):
    pad_x = max(0, patch_size - patch.sizes['x'])
    pad_y = max(0, patch_size - patch.sizes['y'])
    if pad_x > 0 or pad_y > 0:
        padding = ((0, 0), (0, pad_y), (0, pad_x))  # No padding for the band dimension
        for var in patch.data_vars:
            patch[var] = xr.DataArray(
                np.pad(patch[var].values, padding, mode='constant', constant_values=0),
                dims=patch[var].dims,
                coords=patch[var].coords)
    return patch

def update_patch_spatial_ref(patch, i, j, patch_size, original_dataset):
    original_transform = extract_transform(original_dataset)
    original_spatial_ref = original_dataset.spatial_ref.copy()
    x_offset = i * patch_size * original_transform.a
    y_offset = j * patch_size * original_transform.e  # Negative if north-up
    new_transform = Affine(original_transform.a, original_transform.b, original_transform.c + x_offset,
                        original_transform.d, original_transform.e, original_transform.f + y_offset)
    patch = patch.assign_coords({'spatial_ref': original_spatial_ref})
    patch.spatial_ref.attrs['GeoTransform'] = list(new_transform)
    return patch