import rasterio
import geopandas as gpd
from .band import SatelliteBand
from rasterio.features import geometry_mask

# class that has for single image an annotation that probably changes during time -> for sits 
class SatelliteImageAnnotation:
    def __init__(self, shapefile_path):
        self.path = shapefile_path
        self.df = self._load_dataframe()
    	
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
 
    def plot(self):
        return self.band.plot()

class Sentinel2Annotation(SatelliteImageAnnotation):
    def __init__(self, satellite_image, shapefile_path):
        super().__init__(shapefile_path)
        self.satellite_image = satellite_image
        self.band = self._rasterize_annotation()

    def _rasterize_annotation(self):     
        geometries = self.get_geometries_as_list()
        s2_b02_meta = self.satellite_image.meta["B02"]
        s2_b02_meta['dtype'] = 'uint8'
        output_path = self.path.parent / (self.path.stem + ".tif")
        with rasterio.open(output_path, 'w', **s2_b02_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)
        self.band = SatelliteBand(band_name="annotation", band_path=output_path)
        return self.band

class Sentinel1Annotation(SatelliteImageAnnotation):
    def __init__(self, satellite_image, shapefile_path):
        super().__init__(shapefile_path)
        self.satellite_image = satellite_image
        self.band = self._rasterize_annotation()

    def _rasterize_annotation(self, resolution):     
        geometries = self.get_geometries_as_list()
        s1_vv_meta = self.satellite_image.meta["VV"]
        s1_vv_meta['dtype'] = 'uint8'
        output_path = self.path.parent / (self.path.stem + ".tif")
        with rasterio.open(output_path, 'w', **s1_vv_meta) as dst:
            mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
            dst.write(mask.astype(rasterio.uint8), 1)
        self.band = SatelliteBand(band_name="annotation", band_path=output_path)
        return self.band
