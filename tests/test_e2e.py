import pytest
from satellite_datacube.datacube import SatelliteDataCube 
import os
import numpy as np

base_folder = r"D:\SatelliteDataCube\Chimanimani"
test_folder = os.path.join(base_folder, "test_folder")
ts_length = 2
patch_size = 256

@pytest.fixture
def setup_datacube():
    os.makedirs(test_folder, exist_ok=True)
    return SatelliteDataCube(base_folder=base_folder)

def test_initial_loading(setup_datacube):
    assert setup_datacube.satellite_images, "Failed to load satellite images"
    assert setup_datacube.masks, "Failed to load mask for each satellite image"
    assert np.any(setup_datacube.global_mask), "Failed to load global mask"

@pytest.mark.parametrize("ts_length, output_folder", [
    (2, None),
    (4, ""),
    (6, test_folder),
    (2, "not_excisting_folder")
])
def test_create_and_save_timeseries(setup_datacube, ts_length, output_folder):
    timeseries = setup_datacube.create_timeseries(timeseries_length=ts_length)
    setup_datacube.save_timeseries(timeseries, ts_folder=output_folder)
    loaded_timeseries = setup_datacube.load_single_timeseries(timeseries_length=ts_length, ts_folder=output_folder) 
    assert timeseries and len(timeseries) == ts_length, "Failed to create or match timeseries length"
    assert isinstance(timeseries, list), "Failed to return timeseries as type list"
    assert loaded_timeseries, f"Failed to load timeseries with length {ts_length} from folder: {output_folder}"

@pytest.mark.parametrize("patch_size, selected_timeseries", [
    (patch_size, True),
    #(patch_size, False)
])
def test_create_patches(setup_datacube, patch_size, selected_timeseries):
    timeseries = []
    if selected_timeseries:
        timeseries = setup_datacube.create_timeseries(timeseries_length=ts_length) 
    patches = setup_datacube.create_patches(patch_size=patch_size, selected_timeseries=timeseries)
    img, msk, msk_gb = patches.values()	
    assert img.shape[0] == msk.shape[0] == msk_gb.shape[0], "Failed to create same number of patches for the three sources img, msk, msk_gb"
    assert isinstance(patches, dict), "Failed to create patches as dictonary"
    for source, patchArray in patches.items():
        assert patchArray is not None, f"Failed to create patches for source: {source}, patch_size: {patch_size}"
        assert patchArray.shape[-1] == patch_size, f"Failed to create patches with defined patch size: {patch_size}. {patchArray.shape[-1]} should be {patch_size}"
        if source in ["images", "masks"]:
            if timeseries:
                assert patchArray.shape[1] == len(timeseries), f"Length {patchArray.shape[1]}) of created patches is not equal to length of timeseries {len(timeseries)}"
            else:
                assert patchArray.shape[1] == len(setup_datacube.satellite_images), f"Length {patchArray.shape[1]} of created patches is not equal to length of timeseries {len(setup_datacube.satellite_images)}"

@pytest.mark.parametrize("class_values, class_ratio", [
    ([1], (70, 30)),
    ([1], (100, 0)),
    #([1], (70, 70)),
    ([1,2,3], (50, 50)),
])
def test_filter_patches(setup_datacube, class_values, class_ratio):
    timeseries = setup_datacube.create_timeseries(timeseries_length=ts_length) 
    patches = setup_datacube.create_patches(patch_size=patch_size, selected_timeseries=timeseries)
    filtered_patches = setup_datacube.filter_patches(patches, class_values, class_ratio)
    msk_patches = filtered_patches["global_mask"]
    images_of_class = int((np.sum(np.any(msk_patches == 1, axis=(2,3)))/msk_patches.shape[0])*100)
    print(images_of_class, images_of_class==class_ratio[0])
    assert filtered_patches is not None, f"Failed to filter patches with class_values: {class_values}, class_ratio: {class_ratio}"
    assert images_of_class == class_ratio[0], f"Unequal count of images per class: {images_of_class} != {class_ratio[0]}"

@pytest.mark.parametrize("output_folder", [
    (None),
    (""),
    (test_folder),
    ("not_existing_folder"),
])
def test_save_patches(setup_datacube, output_folder):
    timeseries = setup_datacube.load_single_timeseries(timeseries_length=ts_length, ts_folder=test_folder) 
    patches = setup_datacube.create_patches(patch_size=patch_size, selected_timeseries=timeseries)
    filtered_patches = setup_datacube.filter_patches(patches, class_values=[1], class_ratio=(100, 0))
    setup_datacube.save_patches(patches=filtered_patches, patches_folder=output_folder)
    loaded_patches = setup_datacube.load_patches(patch_size=patch_size, timeseries_length=ts_length, patches_folder=output_folder) 
    assert loaded_patches, f"Failed to load patches with size {patch_size} from folder: {output_folder}"

if __name__ == '__main__':
    common_args = [
        "-v", 
        #"-s"
        ]
    tests = [
        "tests\\test_e2e.py::test_initial_loading",
        "tests\\test_e2e.py::test_create_and_save_timeseries",
        "tests\\test_e2e.py::test_create_patches",
        "tests\\test_e2e.py::test_filter_patches",
        "tests\\test_e2e.py::test_save_patches"
    ]
    pytest.main(common_args + tests)
