
from satellite_datacube.datacube_backup import SatelliteDataCube 

# Define folder of satellite data and specify some parameters (read more in the doc-string)
path_of_SITS = r"D:\SatelliteDataCube\Chimanimani" # path_of_SITS = ".../Sentinel2"
satellites = ["S1", "S2"]
dcParameter = { "timeseries_length": 10 , "patch_size": 48}

# Initialisation of the data cube with loading of the data (global_data + patches)
S2_datacube = SatelliteDataCube(base_folder=path_of_SITS, parameters=dcParameter, load_data=True)

# global_mask = S2_datacube.load_global_mask()
# ts16 = S2_datacube.load_single_timeseries(timeseries_length=16)
# ts16 = S2_datacube.create_timeseries(timeseries_length=16, save=True)
# patches128_ts16 = S2_datacube.load_patches_as_dict(patch_size=128, timeseries_length=16)
# patches128_ts16 = S2_datacube.create_patches(
#     source="img", 
#     patch_size=128, 
#     selected_timeseries=[], 
#     indices=False
#     )

# S2_datacube.process_patches(sources=["img","msk","msk_gb"], class_values=[1], seed=42)
# [print(patchArray.shape) for patchArray in S2_datacube.patches.values()]
# S2_datacube.sanity_check()



# We want to use this package for training 

# 