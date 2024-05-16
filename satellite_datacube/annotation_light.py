import rasterio
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
 
class SatCubeAnnotation:
    def __init__(self, inventory_dir):
        self.inventory_dir = Path(inventory_dir)
        self.folder = self.inventory_dir / "annotations"
        self.name = f"{inventory_dir.name}" 
        self.shp_file = self.folder / f"{self.name}.shp"
        self.mask_path = self.folder / f"{self.name}_mask.tif"
        self.gdf = self._initalize_dataframe()
        
    def _initalize_dataframe(self):
        
        def repair_geometry(geom):
            if geom is None or not geom.is_valid:
                return geom.buffer(0) if geom else None
            return geom

        if self.shp_file.exists():
            gdf = gpd.read_file(self.shp_file)
            gdf["geometry"] = gdf["geometry"].apply(repair_geometry)
            return gdf
        else:
            print("No shapefile found in the annotations folder.")
            return None
    
    def rasterize_annotations(self, raster_meta):
        raster_crs = raster_meta["crs"]
        if self.gdf.crs != raster_crs:
            self.gdf.to_crs(raster_crs, inplace=True)

        if not self.mask_path.exists():
            geometries = self.gdf["geometry"].values
            raster_meta.update({'dtype': 'uint8', 'count': 1})

            with rasterio.open(self.mask_path, 'w', **raster_meta) as dst:
                mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
                dst.write(mask.astype(rasterio.uint8), 1)
        else:
            print("Mask file already exists in the annotations folder.")

        return self.mask_path
