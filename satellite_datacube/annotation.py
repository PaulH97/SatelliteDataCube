import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from .utils import pad_patch, get_metadata_of_window
from rasterio.windows import Window
 
class SatelliteImageAnnotation:
    def __init__(self, satellite_image, shapefile_path):
        self.image = satellite_image
        self.shp_path = shapefile_path
        self.mask_path = None
        self.dataframe = None
        
    def load_dataframe(self):
        if self.dataframe is not None:
            return self.dataframe
        df = gpd.read_file(self.shp_path) 
        geometries = df["geometry"].to_list()
        repaired_geometries = []
        for geom in geometries:
            if geom and not geom.is_valid:
                geom = geom.buffer(0)
            repaired_geometries.append(geom)
        df["geometry"] = repaired_geometries
        self.dataframe = df  
        return self.dataframe

    def load_and_transform_to_crs(self, epsg_code):
        """
        Transforms the loaded dataframe to the given CRS specified by an EPSG code.
        """
        annotation_df = self.load_dataframe()  
        target_crs = rasterio.crs.CRS.from_epsg(epsg_code)
        # Check if the dataframe's CRS differs from the target CRS
        if annotation_df.crs != target_crs:
            annotation_df = annotation_df.to_crs(target_crs)
        return annotation_df
    
    def rasterize(self, resolution):
        band_with_desired_res = self.image.find_band_by_res(resolution)
        if band_with_desired_res is None:
            raise ValueError(f"Resolution {resolution} not found. Please refine your resolution to match one of the available image bands.")
        with rasterio.open(band_with_desired_res.path) as src:
            crs_epsg = src.crs.to_epsg()
            raster_meta = src.meta.copy()
            raster_meta.update({'dtype': 'uint8', 'count': 1})

        geometries = self.load_and_transform_to_crs(crs_epsg)["geometry"].to_list()

        mask_filepath = self.image.folder / f"{self.image.name}_{resolution}m_mask_.tif" # S2_mspc_l2a_20190509_10m_mask.tif
        with rasterio.open(mask_filepath, 'w', **raster_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)
        self.mask_path = mask_filepath
        return self
    
    def plot(self):
        return self.band.plot()
    
    def create_patches(self, patch_size, overlay=0, padding=True):
        with rasterio.open(self.mask_path) as src:
            step_size = patch_size - overlay
            for i in range(0, src.width,  step_size):
                for j in range(0, src.height, step_size):
                    window = Window(j, i, patch_size, patch_size)
                    patch = src.read(window=window)        
                    # Check if patch needs padding
                    if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        patch = pad_patch(patch, patch_size)
                    elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
                        continue  # Skip patches that are smaller than patch_size when padding is False
                    # Update metadata for the patch
                    patch_meta = get_metadata_of_window(src, window)
                    patches_folder = self.image.folder / "patches" / "MSK"
                    patches_folder.mkdir(parents=True, exist_ok=True)
                    patch_filepath = patches_folder / f"patch_{i}_{j}.tif"
                    with rasterio.open(patch_filepath, 'w', **patch_meta) as dst:
                            dst.write(patch)
        return patches_folder

# class Sentinel2Annotation(SatelliteImageAnnotation):
#     def __init__(self, shapefile_path):
#         super().__init__(shapefile_path)

#     def _rasterize_annotation(self, band_meta):     
#         geometries = self.get_geometries_as_list()
#         annotation_meta = band_meta
#         annotation_meta['dtype'] = 'uint8'
#         output_path = self.path.parent / (self.path.stem + ".tif")
#         with rasterio.open(output_path, 'w', **annotation_meta) as dst:
#             mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
#             dst.write(mask.astype(rasterio.uint8), 1)
#         self.band = SatelliteBand(band_name="annotation", band_path=output_path)
#         return self.band

# class Sentinel1Annotation(SatelliteImageAnnotation):
#     def __init__(self, satellite_image, shapefile_path):
#         super().__init__(shapefile_path)
#         self.satellite_image = satellite_image
#         self.band = self._rasterize_annotation()

#     def _rasterize_annotation(self, resolution):     
#         geometries = self.get_geometries_as_list()
#         s1_vv_meta = self.satellite_image.meta["VV"]
#         s1_vv_meta['dtype'] = 'uint8'
#         output_path = self.path.parent / (self.path.stem + ".tif")
#         with rasterio.open(output_path, 'w', **s1_vv_meta) as dst:
#             mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
#             dst.write(mask.astype(rasterio.uint8), 1)
#         self.band = SatelliteBand(band_name="annotation", band_path=output_path)
#         return self.band
