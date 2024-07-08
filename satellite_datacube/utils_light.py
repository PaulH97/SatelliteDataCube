import re
from rasterio.enums import Resampling
import os
from pathlib import Path
import pandas as pd
from affine import Affine
import numpy as np 
import xarray as xr
import rioxarray as rxr
from dask.distributed import Client
from typing import Tuple, List
from dask_jobqueue import SLURMCluster
import dask
from contextlib import contextmanager
import signal
from xarray.core.dataarray import DataArray
from dask import delayed
import ruptures as rpt
from matplotlib import pyplot as plt
from memory_profiler import profile
import dask.dataframe as dd
import geopandas as gpd


# Function to load a single file with rioxarray
def load_file(file):
    data = rxr.open_rasterio(file, chunks="auto")

    time = data.attrs['time']
    data_arrays = []

    for i, long_name in enumerate(data.attrs['long_name']):
        band_data = data.sel(band=i+1)
        band_data = band_data.squeeze(drop=True).drop_vars("band")
        band_data = band_data.expand_dims({'time': [time]})
        
        new_da = xr.DataArray(
            band_data,coords={'time': [time],'y': band_data['y'],'x': band_data['x']},
            dims=['time', 'y', 'x'],
            name=long_name
        )
        data_arrays.append(new_da)
    ds = xr.Dataset({da.name: da for da in data_arrays})
    ds.attrs["spatial_ref"] = data.spatial_ref.copy()
    return ds

def calculate_mean_sar_data(vv_data, vh_data):
    if np.isnan(vv_data).all() or np.isnan(vh_data).all():
        return {"VV": np.nan, "VH": np.nan}
    return {"VV": float(np.nanmean(vv_data)), "VH": float(np.nanmean(vh_data))}

def linear_to_db(linear_array):
    """
    Convert linear scale Sentinel-1 data back to decibel (dB) scale.

    Parameters:
    linear_array (np.ndarray): The array of linear scale values.

    Returns:
    np.ndarray: The array of dB values.
    """
    # Ensure no log of zero
    linear_array = np.where(linear_array <= 0, np.nan, linear_array)
    
    # Convert to dB
    db_array = 10 * np.log10(linear_array)
    
    return db_array

def extract_metadata(raster_path): 
    with rioxarray.open_rasterio(raster_path) as raster:
        crs_wkt = raster.rio.crs.to_wkt()
        transform = raster.rio.transform()
        return {"crs_wkt": crs_wkt, "GeoTransform": transform}

def build_xr_dataset(data_array, band_names, date, metadata):

    ds = data_array.to_dataset(name='band_data')
    
    crs_wkt = metadata['crs_wkt']
    geo_transform = metadata['GeoTransform']
    date_np = np.datetime64(date).astype('datetime64[ns]')
    ds = ds.expand_dims(time=[date_np])
    
    for i, band_name in enumerate(band_names):
        ds[band_name] = ds['band_data'].isel(band=i)
    
    ds = ds.drop_vars('band_data')
    ds = ds.drop_vars('band')
    
    ds.rio.write_crs(crs_wkt, inplace=True)
    
    transform_values = list(map(float, geo_transform.split()))
    affine_transform = Affine(*transform_values[:6])
    ds.rio.write_transform(affine_transform, inplace=True)
    
    return ds

def plot_buffer(buffer1, buffer2):
    # Step 3: Extract coordinates for plotting
    x1, y1 = buffer1.exterior.xy
    x2, y2 = buffer2.exterior.xy
    #   Step 4: Plotting using Matplotlib
    plt.figure()
    plt.plot(x1, y1)
    plt.fill(x1, y1, alpha=0.5)  # Fill the polygon with a semi-transparent fill
    plt.plot(x2, y2)
    plt.fill(x2, y2, alpha=0.5)  # Fill the polygon with a semi-transparent fill
    plt.title('Shapely Polygon Plot')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()
    plt.savefig('buffer.png')

def plot_cdndvi(ndvi_df):
    plt.figure(figsize=(10, 6))
    plt.plot(ndvi_df.index, ndvi_df["CDNDVI"], marker='o', linestyle='-')
    plt.title('CDNDVI Over Time')
    plt.xlabel('Date')
    plt.ylabel('CDNDVI')
    plt.grid(True)
    plt.savefig('cdndvi_plot.png')

def despike_timeseries(data, window=5, z_threshold=2):
    """
    Removes spikes from the provided NDVI timeseries data using a moving median and Z-score thresholding.

    Parameters:
    - data (pd.Series): A pandas Series with NDVI values.
    - window (int): The size of the moving window for the median calculation.
    - z_threshold (int or float): The Z-score threshold to identify spikes.

    Returns:
    - pd.Series: The despiked NDVI timeseries.
    """
    # Calculate the moving median and absolute deviation from the median
    rolling_median = data.rolling(window, center=True, min_periods=1).median()
    deviation = np.abs(data - rolling_median)
    median_deviation = deviation.rolling(window, center=True, min_periods=1).median()

    # Replace with NaN where the deviation is too large (i.e., beyond the threshold of median deviation)
    z_score = 0.6745 * deviation / median_deviation
    data_clean = data.copy()
    data_clean[z_score > z_threshold] = rolling_median[z_score > z_threshold]

    return data_clean

def calculate_cdndvi(ndvi_df, window_size=5):
    # Ensure the column names match those in the DataFrame
    smoothed_ndvi = ndvi_df["NDVI"].rolling(window=window_size, center=True, min_periods=1).mean()
    smoothed_undist_ndvi = ndvi_df["NDVI_UNDIST"].rolling(window=window_size, center=True, min_periods=1).mean()
    cdndvi = smoothed_undist_ndvi - smoothed_ndvi
    return cdndvi

def find_significant_changes_v2(ann_ndvi_df, look_back=4, look_ahead=3, threshold_multiplier=2):
    
    ann_ndvi_df['DatesNumeric'] = (ann_ndvi_df.index - ann_ndvi_df.index.min()).days
    slopes = np.diff(ann_ndvi_df['CDNDVI']) / np.diff(ann_ndvi_df['DatesNumeric'])
    slopes_df = pd.Series(slopes, index=ann_ndvi_df.index[:-1])

    # Calculate rolling statistics
    rolling_mean = slopes_df.rolling(window=look_back).mean()
    rolling_std = slopes_df.rolling(window=look_back).std()

    significant_points = []
    # Compare each point to its rolling statistics
    for i in range(len(slopes)):
        if i >= look_back and i < len(slopes) - look_ahead:
            if pd.notna(rolling_mean[i]) and pd.notna(rolling_std[i]):
                if slopes[i] > rolling_mean[i] + threshold_multiplier * rolling_std[i]:
                    if all(slopes[i + j] > rolling_mean[i] for j in range(1, look_ahead + 1)):
                        significant_points.append(ann_ndvi_df.index[i])
                        
    return significant_points

def detect_change_points(cdndvi, penalty_value):

    model = "l1"  # "l1" norm cost (mean absolute deviation), which can be more robust to outliers
    algo = rpt.Binseg(model=model).fit(cdndvi.array)
    
    # Predict change points, 'pen' controls the number of breakpoints
    breakpoints = algo.predict(pen=penalty_value)

    change_points_dates = [cdndvi.index[bkpt - 1] for bkpt in breakpoints if bkpt < len(data)]
    
    return change_points_dates

# def extract_ndvi_values(ann, xds, ann_gdf_crs):
#     time_mask = xds["SCL"].rio.clip([ann.geometry], crs=ann_gdf_crs, drop=True).isin([4,5,6]).mean(dim=("x", "y")) >= 0.5
#     ann_ndvi = xds["NDVI"].rio.clip([ann.geometry], crs=ann_gdf_crs, drop=True).sel(time=time_mask).mean(dim=("x", "y"))
#     ann_geom_buffer = create_differential_buffer(ann.geometry, large_buffer_size=200, small_buffer_size=10)
#     ann_scl_buffer = xds["SCL"].rio.clip([ann_geom_buffer], crs=ann_gdf_crs, drop=True).sel(time=time_mask)
#     ann_ndvi_buffer = xds["NDVI"].rio.clip([ann_geom_buffer], crs=ann_gdf_crs, drop=True).sel(time=time_mask).where(ann_scl_buffer == 4).mean(dim=("x", "y"))
#     ann_ndvi_df = pd.DataFrame({"NDVI": ann_ndvi.values, "NDVI_UNDIST": ann_ndvi_buffer.values}, index=ann_ndvi.time.values)
#     return ann_ndvi_df

@dask.delayed
def clip_dataset(geometry, xds, crs):
    return xds.rio.clip([geometry], crs=crs, all_touched=True, drop=True, from_disk=True)

def create_time_mask(xds, valid_scl_keys=[4,5,6]): 
    valid_values_mask = xds["SCL"].isin(valid_scl_keys)
    non_nan_mask = ~np.isnan(xds["SCL"])
    combined_mask = valid_values_mask & non_nan_mask
    valid_count = non_nan_mask.sum(dim=("x", "y"))
    condition_count = combined_mask.sum(dim=("x", "y"))
    time_mask = (condition_count / valid_count) > 0.5
    return time_mask

@dask.delayed
def compute_ndvi(clipped_xds, time_mask):
    return clipped_xds["NDVI"].sel(time=time_mask).mean(dim=("x", "y"))

@dask.delayed
def compute_buffer_ndvi(xds, buffer_geometry, crs, time_mask):
    clipped_buffer_xds = xds.rio.clip([buffer_geometry], crs=crs, all_touched=True, drop=True, from_disk=True)
    ann_scl_buffer = clipped_buffer_xds["SCL"].sel(time=time_mask)
    ann_ndvi_buffer = clipped_buffer_xds["NDVI"].sel(time=time_mask).where(ann_scl_buffer == 4).mean(dim=("x", "y"), skipna=True)
    return ann_ndvi_buffer

@dask.delayed
def create_dataframe(ann_ndvi, ann_ndvi_buffer, time_values):
    return pd.DataFrame({"NDVI": ann_ndvi, "NDVI_UNDIST": ann_ndvi_buffer}, index=time_values)

@dask.delayed
def extract_ndvi_buffer(ann, xds):
    ann_geom_buffer = create_differential_buffer(ann.geometry, large_buffer_size=200, small_buffer_size=10)
    xds_buffer = xds.rio.clip([ann_geom_buffer], crs=xds.rio.crs, all_touched=True, drop=True, from_disk=True)
    ann_ndvi_buffer = xds_buffer["NDVI"].where(xds_buffer["SCL"] == 4).mean(dim=("x", "y"), skipna=True)
    return ann_ndvi_buffer

@dask.delayed
def extract_ndvi(ann: pd.Series, xds: xr.Dataset) -> pd.DataFrame:
    clipped_xds = xds.rio.clip([ann.geometry], crs=xds.rio.crs, all_touched=True, drop=True, from_disk=True)
    ann_ndvi = clipped_xds["NDVI"].mean(dim=("x", "y"))
    return ann_ndvi

def extract_ndvi_values(ann, xds):
    crs = xds.rio.crs
    clipped_xds = clip_dataset(ann.geometry, xds, crs)
    time_mask = create_time_mask(clipped_xds, valid_scl_keys=[4, 5, 6])
    ann_ndvi = compute_ndvi(clipped_xds, time_mask)
    
    ann_geom_buffer = create_differential_buffer(ann.geometry, large_buffer_size=200, small_buffer_size=10)
    ann_ndvi_buffer = compute_buffer_ndvi(xds, ann_geom_buffer, crs, time_mask)
    
    ann_ndvi_df = create_dataframe(ann_ndvi, ann_ndvi_buffer, ann_ndvi.time)
    
    return clipped_xds

def extract_ndvi_values(ann, xds):
    clipped_xds = xds.rio.clip([ann.geometry], crs=xds.rio.crs, all_touched=True, drop=True)
    time_mask = create_time_mask(clipped_xds, valid_scl_keys=[4, 5, 6])
    if time_mask.values:
        ann_ndvi = clipped_xds["NDVI"].mean(dim=("x", "y"), skipna=True).item()
        ann_geom_buffer = create_differential_buffer(ann.geometry, large_buffer_size=200, small_buffer_size=10)
        clipped_buffer_xds = xds.rio.clip([ann_geom_buffer], crs=xds.rio.crs, all_touched=True, drop=True)
        ann_ndvi_buffer = clipped_buffer_xds["NDVI"].where(clipped_buffer_xds["SCL"] == 4).mean(dim=("x", "y"), skipna=True).item()
        return {"ID": ann["id"], "NDVI": ann_ndvi, "NDVI_UNDIST": ann_ndvi_buffer,"Date": clipped_xds.time.values[0]}
    else:
        return {"ID": ann["id"], "NDVI": None, "NDVI_UNDIST": None,"Date": None}

def process_image_and_extract_ndvi(s2, s2_ann: gpd.GeoDataFrame):
    if s2.path.exists():
        xds = s2.load_data()
        if "NDVI" not in xds:
            xds = s2.calculate_ndvi()
        # For each image dataset, extract NDVI values for all annotations
        ndvi_results = [extract_ndvi_values(ann, xds) for _, ann in s2_ann.iterrows()]
        return ndvi_results
    return []

# @dask.delayed
# def find_ndvi_windows(ann, xds, ann_gdf_crs, window_size: int = 5):
#     ann_ndvi_df = extract_ndvi_values(ann, xds, ann_gdf_crs)
#     ann_ndvi_df['CDNDVI'] = calculate_cdndvi(ann_ndvi_df, window_size=window_size)
#     change_points_dates = detect_change_points(cdndvi=ann_ndvi_df['CDNDVI'], penalty_value=1)
#     # for change_point in change_points_dates:
#     #     start_date = change_point - pd.DateOffset(months=3)
#     #     end_date = change_point + pd.DateOffset(months=3)
#     #     windowed_data = ann_ndvi_df.loc[start_date:end_date]
#     #     windowed_data['Despiked_NDVI'] = despike_timeseries(windowed_data['NDVI'], window_size=window_size, z_threshold=2)
#     return change_points_dates

def create_differential_buffer(geometry, large_buffer_size=200, small_buffer_size=10):
    large_buffer = geometry.buffer(large_buffer_size)
    small_buffer = geometry.buffer(small_buffer_size)
    differential_buffer = large_buffer.difference(small_buffer)
    return differential_buffer

def classify_patch(i: int, j: int, xds: xr.Dataset, patch_x_end: int, patch_y_end: int) -> Tuple[str, Tuple[int, int]]:
    patch = xds.isel(x=slice(i, patch_x_end), y=slice(j, patch_y_end))
    if patch.MASK.values.sum() != 0:
        return "Annotation", (i, j)
    else:
        return "No-Annotation", (i, j)

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
def update_patch_spatial_ref(patch, i, j, patch_size, xds):
    original_transform = extract_transform(xds)
    x_offset = i * patch_size * original_transform.a
    y_offset = j * patch_size * original_transform.e  # Negative if north-up
    new_transform = Affine(original_transform.a, original_transform.b, original_transform.c + x_offset,
                        original_transform.d, original_transform.e, original_transform.f + y_offset)
    patch.attrs['GeoTransform'] = list(new_transform)
    patch.rio.write_crs(xds.rio.crs, inplace=True)
    return patch

def process_patch(patch_idx:List[Tuple], xds:xr.Dataset, patch_size:int, patch_folder:Path) -> Path:
    i, j = patch_idx
    patch = xds.isel(x=slice(i, min(i + patch_size, xds.x.size)),y=slice(j, min(j + patch_size, xds.y.size)))
    # pad patch with .rio.pad_box
        
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

def resample_raster(raster, target_resolution=10):
    current_resolution = raster.rio.resolution()
    if current_resolution != (target_resolution, target_resolution):
        scale_factor = current_resolution[0] / target_resolution
        new_shape = (int(raster.rio.shape[0] * scale_factor), int(raster.rio.shape[1] * scale_factor))
        resampled_raster = raster.rio.reproject(raster.rio.crs, shape=new_shape, resampling=Resampling.bilinear)
        return resampled_raster
    else:
        return raster
    
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

# def S2_xds_preprocess(ds):
#     # Extract the date from the file path and set as a time coordinate
#     date_str = Path(ds.encoding['source']).parent.stem
#     date_np = np.datetime64(pd.to_datetime(date_str)).astype('datetime64[ns]')
#     ds = ds.expand_dims(time=[date_np])

#     band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
#     for i, band_name in enumerate(band_names):
#         ds[band_name] = ds['band_data'].isel(band=i)
    
#     # Drop the original multi-band data variable if it is no longer needed
#     ds = ds.drop_vars('band_data')

#     return ds

# def S1_xds_preprocess(ds):
#     # Extract the date from the file path and set as a time coordinate
#     date_str = Path(ds.encoding['source']).parent.stem
#     date_np = np.datetime64(pd.to_datetime(date_str)).astype('datetime64[ns]')
#     ds = ds.expand_dims(time=[date_np])

#     band_names = ["VH", "VV"]
#     for i, band_name in enumerate(band_names):
#         ds[band_name] = ds['band_data'].isel(band=i)
    
#     ds = ds.drop_vars('band_data')
#     return ds

def extract_transform(xds):
    new_order_indices = [1, 2, 0, 4, 5, 3] 
    original_transform = xds.spatial_ref.attrs['GeoTransform'].split(" ")
    original_transform = [float(val) for val in original_transform]
    original_transform = [original_transform[i] for i in new_order_indices]
    return Affine(*original_transform)