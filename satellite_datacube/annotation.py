import rasterio
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
from .utils import pad_patch, get_metadata_of_window
from rasterio.windows import Window

class SatelliteImageAnnotation:
    def __init__(self, satellite_image, shapefile_path):
        self.image = satellite_image
        self.shp_path = Path(shapefile_path)
        self.mask_path = self._find_valid_mask_file()
        self.dataframe = self._load_dataframe()

    def _find_valid_mask_file(self, resolution=10):
        mask_suffixes = ['.tif', '.tiff']
        mask_keyword = f"{resolution}m_mask" if resolution else "mask"
        files = [file for file in self.image.folder.iterdir() if file.is_file() and file.suffix.lower() in mask_suffixes and mask_keyword in file.stem.lower()]
        return files[0] if files else None

    def _load_dataframe(self):
        df = gpd.read_file(self.shp_path)
        df["geometry"] = df["geometry"].apply(self._repair_geometry)
        return df

    @staticmethod
    def _repair_geometry(geom):
        if geom is None or not geom.is_valid:
            return geom.buffer(0) if geom else None
        return geom

    def transform_annotations_to_image_crs(self):
        with rasterio.open(self.image.path) as src:
            image_crs = src.crs
        if self.dataframe.crs != image_crs:
            self.dataframe.to_crs(image_crs, inplace=True)

    def rasterize_annotations(self, resolution):
        if not self.mask_path:
            self._create_mask(resolution)
        return self.mask_path

    def _create_mask(self, resolution):
        self.transform_annotations_to_image_crs()
        band_path = self.image.find_band_by_resolution(resolution)
        if band_path is None:
            raise ValueError(f"Resolution {resolution} not found in image bands.")

        with rasterio.open(band_path) as src:
            raster_meta = src.meta.copy()
            raster_meta.update({'dtype': 'uint8', 'count': 1})

        geometries = self.dataframe["geometry"].values
        mask_filepath = self.image.folder / f"{self.image.name}_{resolution}m_mask.tif"

        with rasterio.open(mask_filepath, 'w', **raster_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)

        self.mask_path = mask_filepath

    def create_patches(self, patch_size, overlay=0, padding=True, output_dir=None):
        output_dir = Path(output_dir) if output_dir else self.image.folder / "patches" / "MSK"
        output_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.mask_path) as src:
            patches_folder = self._generate_patches(src, patch_size, overlay, padding, output_dir)
        return patches_folder

    def _generate_patches(self, src, patch_size, overlay, padding, output_dir):
        step_size = patch_size - overlay
        for i in range(0, src.width, step_size):
            for j in range(0, src.height, step_size):
                patch, window = self._extract_patch(src, i, j, patch_size, padding)
                if patch is not None:
                    self._save_patch(patch, window, src, i, j, output_dir)
        return output_dir

    @staticmethod
    def _extract_patch(src, i, j, patch_size, padding):
        window = Window(i, j, patch_size, patch_size)
        patch = src.read(window=window)
        if padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
            patch = np.pad(patch, ((0, 0), (0, patch_size - patch.shape[1]), (0, patch_size - patch.shape[2])), mode='constant')
        elif not padding and (patch.shape[1] != patch_size or patch.shape[2] != patch_size):
            return None, None
        return patch, window
    
    def _save_patch(self, patch, window, src, i, j, output_dir):
        patch_meta = src.meta.copy()
        patch_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        patch_filename = f"patch_{i}_{j}.tif"
        patch_filepath = output_dir / patch_filename
        
        with rasterio.open(patch_filepath, 'w', **patch_meta) as patch_dst:
            patch_dst.write(patch)

    # def calculate_scl_data_of_annotation(self):
    #     annotation_scl = {}
    #     annotation_df = self.annotation.load_dataframe()
    #     for index, row in annotation_df.iterrows():
    #         with rasterio.open(self.scl_path) as src:
    #             out_image, out_transform = mask(src, [row["geometry"]], crop=True)
    #             annotation_scl[row["id"]] = out_image.flatten()
    #     return annotation_scl

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
