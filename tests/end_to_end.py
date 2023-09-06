from satellite_datacube.datacube import SatelliteDataCube 
import numpy as np

# End-to-End Tests for the SatelliteDataCube loading class
def test_initial_loading():
    S2_datacube = SatelliteDataCube(base_folder=r"D:\SatelliteDataCube\Chimanimani")
    assert S2_datacube.satellite_images is not None, "Failed to load satellite images"
    assert S2_datacube.masks is not None, "Failed to load mask for each satellite image"
    assert S2_datacube.global_mask is not None, "Failed to load global mask"

def test_load_satellite_images():
    S2_datacube = SatelliteDataCube(base_folder=r"D:\SatelliteDataCube\Chimanimani")
    images = S2_datacube.load_satellite_images()
    assert images is not None, "Failed to load satellite images"

def test_load_masks():
    S2_datacube = SatelliteDataCube(base_folder=r"D:\SatelliteDataCube\Chimanimani")
    masks = S2_datacube.load_masks()
    assert masks is not None, "Failed to load masks"   

def test_load_global_mask():
    S2_datacube = SatelliteDataCube(base_folder=r"D:\SatelliteDataCube\Chimanimani")
    global_mask = S2_datacube.load_global_mask()
    assert global_mask is not None, "Failed to load masks"   

def test_load_single_timeseries():
    S2_datacube = SatelliteDataCube(base_folder=r"D:\SatelliteDataCube\Chimanimani")
    timeseries_length = 10
    timeseries = S2_datacube.load_single_timeseries(timeseries_length=timeseries_length)
    assert timeseries is not None, "Failed to load masks"
    assert len(timeseries) == timeseries_length, "Failes to load timeseries with defined length"
  
def run_data_loading_tests():
    print("Testing initial data loading")
    test_initial_loading()
    print("Testing load_global_mask()")
    test_load_global_mask()
    print("Testing load_masks()")
    test_load_masks()
    print("Testing load_satellite_images()")
    test_load_satellite_images()
    print("Testing load_single_timeseries()")
    test_load_masks()
    print("All tests passed!")

if __name__ == '__main__':
    run_data_loading_tests()

