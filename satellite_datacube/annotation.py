import rasterio
from rasterio.mask import mask
import geopandas as gpd
from .band import SatelliteBand
from rasterio.features import geometry_mask

class Annotation:
    def __init__(self, shapefile_path):
        self.path = shapefile_path
        self.df = self._load_dataframe()
        self.band = None
    	
    def _load_dataframe(self):
        return gpd.read_file(self.path) 

    def _repair_geometries(self):
        geometries = self.df["geometry"].to_list()
        repaired_geometries = []
        for geom in geometries:
            if geom:
                if geom.is_valid:
                    repaired_geometries.append(geom)
                else:
                    repaired_geometries.append(geom.buffer(0))

        self.df["geometry"] = repaired_geometries
        return  
    
    def get_geometries_as_list(self):
        self._repair_geometries()
        return self.df["geometry"].to_list()

    def rasterize_annotation(self, raster_metadata):     
        geometries = self.get_geometries_as_list()
        raster_metadata['dtype'] = 'uint8'
        output_path = self.path.parent / (self.path.stem + ".tif")
        with rasterio.open(output_path, 'w', **raster_metadata) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)

        self.band = SatelliteBand(band_name="annotation", band_path=output_path)
        return self.band

    def plot(self):
        return self.band.plot()
