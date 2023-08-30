
from satellite_datacube.datacube import SatelliteDataCube 

# Define folder of satellite data and specify some parameters (read more in the doc-string)
path_of_SITS = r"D:\SatelliteDataCube\Chimanimani" # path_of_SITS = ".../Sentinel2"
satellites = ["S1", "S2"]
dcParameter = { "timeseries_length": 4 , "patch_size": 128}

# Initialisation of the data cube with loading of the data (global_data + patches)
S2_datacube = SatelliteDataCube(base_folder=path_of_SITS, parameters=dcParameter, load_data=True)
# S2_datacube.process_patches(sources=["img", "msk", "msk_gb"], class_values=[1], seed=42)
[print(patchArray.shape) for patchArray in S2_datacube.patches.values()]