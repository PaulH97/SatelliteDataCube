import re
from rasterio.enums import Resampling
import os
from pathlib import Path
import pandas as pd
from affine import Affine
import numpy as np 
import xarray as xr
import rioxarray
from dask.distributed import Client
from typing import Tuple, List
from dask_jobqueue import SLURMCluster
import dask
from contextlib import contextmanager
import signal
from xarray.core.dataarray import DataArray
from dask import delayed
from rasterio.features import geometry_mask
def geotransform_to_affine(geotransform_str):
    """Convert a GeoTransform string to an Affine transformation object."""
    parts = list(map(float, geotransform_str.split()))
    if len(parts) != 6:
        raise ValueError("GeoTransform string must have exactly 6 components.")
    return Affine(parts[1], parts[2], parts[0], parts[4], parts[5], parts[3])

def create_mask(geometry, buffer_dist, transform, shape):
    """Create a mask for the given geometry and buffer distance."""
    return geometry_mask([geometry.buffer(buffer_dist)], transform=transform, invert=True, out_shape=shape)

def calculate_good_pixel_mask(scl_data, bad_pixel_values):
    """Calculate the good pixel mask for the entire dataset."""
    bad_pixel_mask = scl_data.isin(bad_pixel_values)
    good_pixel_mask = ~bad_pixel_mask
    return good_pixel_mask

def extract_ndvi_for_dating(ann, xds, min_good_pixel_ratio=0.7):
    bad_pixel_values = [0, 1, 2, 3, 8, 9, 10, 11]
    geometry = ann['geometry']
    xform = geotransform_to_affine(xds.spatial_ref.attrs['GeoTransform'])
    shape = (xds.dims['y'], xds.dims['x'])

    # Create mask for the annotation
    ann_mask = create_mask(geometry, 0, xform, shape)
    ann_mask_da = xr.DataArray(ann_mask, dims=['y', 'x'])

    # Calculate good pixel mask for the entire dataset
    good_pixel_mask = calculate_good_pixel_mask(xds['SCL'], bad_pixel_values)

    # Prepare results
    results = []
    for time_slice in xds.time:
        print(f"Processing time slice: {time_slice.values}")

        # Select the good pixel mask for the current time slice
        good_pixel_ratio = good_pixel_mask.sel(time=time_slice).mean(dim=['y', 'x']).compute()
        print(f"Good pixel ratio: {good_pixel_ratio.values}")

        if good_pixel_ratio >= min_good_pixel_ratio:
            # Calculate NDVI mean for annotation for the current time slice
            ndvi_time = xds['NDVI'].sel(time=time_slice)
            ann_ndvi_mean = calculate_ndvi_mean(ndvi_time, ann_mask_da).compute()

            result = {
                'time': np.datetime_as_string(time_slice.values, unit='s'),
                'annotation_id': ann.get('id', 'N/A'),
                'annotation_ndvi_mean': float(ann_ndvi_mean.values),
            }
            results.append(result)

    return results
def calculate_ndvi_mean(ndvi_data, mask):
    """Apply mask to NDVI data and calculate mean."""
    filtered_ndvi = ndvi_data.where(mask, drop=True)
    return filtered_ndvi.mean(dim=['y', 'x'])


def classify_patch(i: int, j: int, xds: xr.Dataset, patch_x_end: int, patch_y_end: int) -> Tuple[str, Tuple[int, int]]:
    patch = xds.isel(x=slice(i, patch_x_end), y=slice(j, patch_y_end))
    if patch.MASK.values.sum() != 0:
        return "Annotation", (i, j)
    else:
        return "No-Annotation", (i, j)

@dask.delayed
def set_nan_value(xds, fill_value=-9999):
    # this is not filling the nan - sets a metadata attribute that can be used by downstream tools 
    # to interpret the missing values correctly 
    for var_name in xds.variables:
        if var_name not in xds.coords and var_name not in xds.attrs:          
            xds[var_name].encoding['_FillValue'] = fill_value
    return xds

@dask.delayed
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

@dask.delayed
def update_patch_spatial_ref(patch, i, j, patch_size, original_transform, original_spatial_ref):
    x_offset = i * patch_size * original_transform.a
    y_offset = j * patch_size * original_transform.e  # Negative if north-up
    new_transform = Affine(original_transform.a, original_transform.b, original_transform.c + x_offset,
                        original_transform.d, original_transform.e, original_transform.f + y_offset)
    patch = patch.assign_coords({'spatial_ref': original_spatial_ref})
    patch.spatial_ref.attrs['GeoTransform'] = list(new_transform)
    return patch

def process_patch(patch_idx:List[Tuple], xds:xr.Dataset, patch_size:int, patch_folder:Path) -> Path:
    i, j = patch_idx
    patch = xds.isel(x=slice(i, min(i + patch_size, xds.x.size)),y=slice(j, min(j + patch_size, xds.y.size)))
    patch = pad_patch(patch, patch_size)
    xds_spatial_ref = xds.spatial_ref.copy()
    xds_transform = extract_transform(xds)
    patch = update_patch_spatial_ref(patch, i, j, patch_size, xds_transform, xds_spatial_ref)
    patch = set_nan_value(patch, fill_value=-9999)
    filename = Path(patch_folder, f"patch_{i}_{j}.nc")
    filename.parent.mkdir(parents=True, exist_ok=True)
    # may be necessary to set the environment variable HDF5_USE_FILE_LOCKING=FALSE 
    delayed_obj = patch.to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
    return delayed_obj

def process_patch_chunk(patch_idxs_chunk:List[Tuple], xds:xr.Dataset, xds_transform:Affine, xds_spatial_ref:DataArray, patch_size:int, patch_folder:Path) -> List[Path]:
    tasks = []
    for patch_idx in patch_idxs_chunk:
        i, j = patch_idx
        patch = xds.isel(x=slice(i, min(i + patch_size, xds.x.size)), y=slice(j, min(j + patch_size, xds.y.size)))
        patch = pad_patch(patch, patch_size)
        patch = update_patch_spatial_ref(patch, i, j, patch_size, xds_transform, xds_spatial_ref)
        patch = set_nan_value(patch, fill_value=-9999)  
        
        filename = Path(patch_folder, f"patch_{i}_{j}.nc")
        filename.parent.mkdir(parents=True, exist_ok=True)

        delayed_obj = patch.to_netcdf(filename, format="NETCDF4", engine="netcdf4", compute=False)
        tasks.append(delayed_obj)
    return tasks

def resample_band(band_path, target_resolution=10):
    with rioxarray.open_rasterio(band_path, chunks='auto') as raster:
        current_resolution = raster.rio.resolution()
        if current_resolution != (target_resolution, target_resolution):
            scale_factor = current_resolution[0] / target_resolution
            new_shape = (int(raster.rio.shape[0] * scale_factor), int(raster.rio.shape[1] * scale_factor))
            resampled_raster = raster.rio.reproject(
                raster.rio.crs,
                shape=new_shape,
                resampling=Resampling.bilinear
            )
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

@contextmanager
def setup_dask_client(slurm_config):
    """Setup the Dask client with dynamic worker and thread configuration."""

    # Initialize the SLURM cluster with the configuration parameters
    cluster = SLURMCluster(
        queue=slurm_config['partition'],
        account=slurm_config['account'],
        walltime=slurm_config['walltime'],
        cores=slurm_config['cpus_per_task'],
        memory=slurm_config['memory'],
        log_directory=slurm_config['log_directory'],
        job_cpu=slurm_config['cpus_per_task'],  # Ensure correct CPU allocation
        job_mem=slurm_config['memory'],  # Ensure correct memory allocation
        job_script_prologue=[
            f'export DASK_WORKER_NTHREADS={slurm_config["cpus_per_task"]}',
            f'export DASK_WORKER_MEMORY_LIMIT={slurm_config["memory"]}'
        ],
        job_extra_directives=[
            f"--clusters={slurm_config['clusters']}",
            f"--job-name={slurm_config['job_name']}",
            f"--error={slurm_config['error_file']}",
            f"--output={slurm_config['output_file']}"
        ],
    )

    import pdb; pdb.set_trace()
        
    # Scale the cluster to the desired number of workers - CAREFUL: based on the settings (CPUs + RAM) it can use multiple nodes 
    cluster.scale(jobs=slurm_config['jobs']) # # Request 1 worker 

    client = Client(cluster)

    # Signal handler to ensure cleanup
    def cleanup(signum, frame):
        print("Cleaning up Dask cluster...")
        client.close()
        cluster.close()
        print("Dask cluster cleaned up.")
        exit(0)

    # Register the signal handlers
    signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Handle termination signals
    
    try:
        yield client
    finally:
        # Ensure resources are closed in normal execution
        print("Normal cleanup of Dask cluster...")
        client.close()
        cluster.close()

def normalize(array):
    # Normalize the bands for plotting
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))

def S2_xds_preprocess(ds):
    # Extract the date from the file path and set as a time coordinate
    date_str = Path(ds.encoding['source']).parent.stem
    date_np = np.datetime64(pd.to_datetime(date_str)).astype('datetime64[ns]')
    ds = ds.expand_dims(time=[date_np])

    band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
    for i, band_name in enumerate(band_names):
        ds[band_name] = ds['band_data'].isel(band=i)
    
    # Drop the original multi-band data variable if it is no longer needed
    ds = ds.drop_vars('band_data')

    return ds

def S1_xds_preprocess(ds):
    # Extract the date from the file path and set as a time coordinate
    date_str = Path(ds.encoding['source']).parent.stem
    date_np = np.datetime64(pd.to_datetime(date_str)).astype('datetime64[ns]')
    ds = ds.expand_dims(time=[date_np])

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