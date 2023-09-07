from satellite_datacube.datacube import SatelliteDataCube 
import unittest
from unittest.mock import MagicMock, patch

path_of_SITS = r"D:\SatelliteDataCube\Chimanimani" # path_of_SITS = ".../Sentinel2"
S2_datacube = SatelliteDataCube(base_folder=path_of_SITS)
print(S2_datacube.global_mask)

# class TestDataCube(unittest.TestCase):

#     def setUp(self):
#         self.dc = S2_datacube
#         self.dc.satellite_images = MagicMock()

#     def test_create_patches_no_timeseries(self):
#         """Test create_patches without a specific timeseries."""
        
#         # Mock the process_patches method of each image
#         mock_image = MagicMock()
#         mock_image.process_patches.return_value = ["patch1", "patch2"]
#         self.dc.satellite_images.values.return_value = [mock_image]
        
#         result = self.dc.create_patches(source="img", patch_size=10)
#         # Check that the mock's process_patches method was called once
#         mock_image.process_patches.assert_called_once()
#         self.assertIn("some_source", result)

#     def test_create_patches_with_timeseries(self):
#         """Test create_patches with a specific timeseries."""
        
#         # Mock the process_patches method of each image in the timeseries
#         mock_image1 = MagicMock()
#         mock_image1.process_patches.return_value = ["patch1", "patch2"]
        
#         mock_image2 = MagicMock()
#         mock_image2.process_patches.return_value = ["patch3", "patch4"]
        
#         timeseries = [mock_image1, mock_image2]
        
#         result = self.dc.create_patches(source="another_source", patch_size=20, selected_timeseries=timeseries)
        
#         # Check that the mock's process_patches method was called for both images
#         mock_image1.process_patches.assert_called_once()
#         mock_image2.process_patches.assert_called_once()
#         self.assertIn("another_source", result)

# if __name__ == '__main__':
#     unittest.main()

# # Testing loading of data
# global_mask = 
# patches =
# timeseries = 

# # Testing of loading or creating data 
# global_mask = 
# patches =
# timeseries = 

# # Testing of create_patches()
# # 
# patches =
# patches =
