import rasterio
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
import numpy as np
from rasterio.mask import mask
from satellite_datacube.utils_light import resample_raster, linear_to_db
import rioxarray as rxr
import numpy as np 

np.seterr(divide='ignore', invalid='ignore')
 
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
    
    def rasterize_annotations(self, raster_meta, reset=False):
        if reset and self.mask_path.exists():
            self.mask_path.unlink()
            
        raster_crs = raster_meta["crs"]
        if self.gdf.crs != raster_crs:
            self.gdf.to_crs(raster_crs, inplace=True)

        if not self.mask_path.exists():
            geometries = self.gdf["geometry"].values
            raster_meta.update({'dtype': 'uint8', 'count': 1, "nodata": 255})

            with rasterio.open(self.mask_path, 'w', **raster_meta) as dst:
                mask = geometry_mask(geometries=geometries, invert=True, transform=dst.transform, out_shape=dst.shape)
                dst.write(mask.astype(np.uint8), 1)
        else:
            print("Mask file already exists in the annotations folder.")

        return self.mask_path
    
    def calculate_lst_values(self, s1_t_prev, s1_t_curr):
            
        anns_lst_data = {}

        with rasterio.open(s1_t_prev.path) as s1_t_prev_src, rasterio.open(s1_t_curr.path) as s1_t_curr_src:
            
            if self.gdf.crs != s1_t_prev_src.crs:
                self.gdf = self.gdf.to_crs(s1_t_prev_src.crs)

            for _, row in self.gdf.iterrows():

                s1_t_prev_clip, _ = mask(s1_t_prev_src, [row.geometry.__geo_interface__], crop=True)
                s1_t_curr_clip, _ = mask(s1_t_curr_src, [row.geometry.__geo_interface__], crop=True)

                s1_t_prev_db = linear_to_db(s1_t_prev_clip) # converts nodata value from -9999 to np.nan
                s1_t_curr_db = linear_to_db(s1_t_curr_clip)

                vv_prev, vh_prev = s1_t_prev_db[0], s1_t_prev_db[1]
                vv_curr, vh_curr = s1_t_curr_db[0], s1_t_curr_db[1]
                
                vv_lst = self.calculate_lst(vv_prev, vv_curr)
                vh_lst = self.calculate_lst(vh_prev, vh_curr)
                lst = vv_lst + vh_lst 

                anns_lst_data[row["id"]] = {"LST_VV": vv_lst, "LST_VH": vh_lst, "LST": lst}

        return anns_lst_data
    
    @staticmethod
    def calculate_lst(t_prev, t_curr):
        """
        Calculate the LST index for a given pair of time-step arrays, ignoring nodata values.

        Parameters:
        t_prev (np.ndarray): The array of backscatter values at the previous time step.
        t_curr (np.ndarray): The array of backscatter values at the current time step.
        nodata (float): The value representing nodata in the arrays.

        Returns:
        float: The calculated LST index.
        """
        masked_t_curr = np.ma.masked_array(t_curr, np.isnan(t_curr))
        masked_t_prev = np.ma.masked_array(t_prev, np.isnan(t_prev))

        diff_squared = (masked_t_curr - masked_t_prev) ** 2
        sum_diff_squared = np.sum(diff_squared)
        n = np.ma.count(diff_squared)

        lst_index = sum_diff_squared / n if n > 0 else np.nan
        return lst_index

    def calculate_ndvi_values(self, ndvi_path, scl_path, good_pixel_threshold=70):

        anns_ndvi_data = {}
        annotation_zones = self.preprocess_annotation_zones(buffer_distance=20)

        with rxr.open_rasterio(scl_path) as scl_src:
            scl_resampled = resample_raster(scl_src)
            scl_resampled.rio.to_raster(scl_path)

        with rasterio.open(scl_path) as scl_src, rasterio.open(ndvi_path) as ndvi_src:
            
            if self.gdf.crs != scl_src.crs:
                self.gdf = self.gdf.to_crs(scl_src.crs)
            
            for _, row in self.gdf.iterrows():
            
                annotation_ndvi_mean, annotation_ndvi_median, ann_pixel_count = self.calculate_annotation_ndvi(
                    clip_geom=row.geometry.__geo_interface__, 
                    scl_src=scl_src, 
                    ndvi_src=ndvi_src, 
                    good_pixel_threshold=good_pixel_threshold
                    )
                
                buffer_ndvi_mean, buffer_ndvi_median = self.calculate_annotation_buffer_ndvi(
                    row.geometry, annotation_zones, scl_src, ndvi_src, ann_pixel_count)
                                
                anns_ndvi_data[int(row["id"])] = {"NDVI": annotation_ndvi_mean, "NDVI_undist": buffer_ndvi_mean}

        return anns_ndvi_data

    def calculate_annotation_ndvi(self, clip_geom, scl_src, ndvi_src, good_pixel_threshold):

        bad_pixel_values = [0, 1, 2, 3, 8, 9, 10, 11]
        scl_ann, _ = mask(scl_src, [clip_geom], crop=True, nodata=99)
        ndvi_ann, _ = mask(ndvi_src, [clip_geom], crop=True)
        
        valid_scl_mask = scl_ann != 99
        valid_ndvi_mask = ~np.isnan(ndvi_ann)
        combined_valid_mask = valid_scl_mask & valid_ndvi_mask
        
        good_pixel_mask = (~np.isin(scl_ann[combined_valid_mask], bad_pixel_values))
        valid_ndvi_ann = ndvi_ann[combined_valid_mask].flatten()

        if good_pixel_mask.size != valid_ndvi_ann.size:
            raise ValueError("Mismatch in good pixel mask and valid NDVI data sizes.")
            
        if good_pixel_mask.size != 0:
            good_pixel_percentage = (np.sum(good_pixel_mask) / good_pixel_mask.size) * 100.0
        else:
            good_pixel_percentage = 0
            
        if good_pixel_percentage >= good_pixel_threshold:
            ndvi_masked_ann = valid_ndvi_ann[good_pixel_mask]

            if ndvi_masked_ann.size > 0:
                if np.isnan(ndvi_masked_ann).all():
                    ndvi_mean_ann = ndvi_median_ann = None
                else:
                    ndvi_mean_ann = np.nanmean(ndvi_masked_ann)
                    ndvi_median_ann = np.nanmedian(ndvi_masked_ann)
            else:
                ndvi_mean_ann = ndvi_median_ann = None
        else:
            ndvi_mean_ann = ndvi_median_ann = None

        ann_pixel_count = np.sum(combined_valid_mask)

        return ndvi_mean_ann, ndvi_median_ann, ann_pixel_count
    
    def calculate_annotation_buffer_ndvi(self, geometry, annotation_zones, scl_src, ndvi_src, ann_pixel_count):
        buffer_dist = 100  # Starting buffer distance
        num_of_pixels = ann_pixel_count * 5
        while True:
            ann_large_buffer = geometry.buffer(buffer_dist)
            buffered_zone = ann_large_buffer
            for zone in annotation_zones:
                buffered_zone = ann_large_buffer.difference(zone)
            scl_data, _ = mask(scl_src, [buffered_zone], crop=True)
            ndvi_data, _ = mask(ndvi_src, [buffered_zone], crop=True)
            vegetation_mask = scl_data[0] == 4  # Targets only vegetation pixels
            ndvi_masked = ndvi_data[0][vegetation_mask].flatten()

            if ndvi_masked.size > 0:
                ndvi_masked = ndvi_masked[~np.isnan(ndvi_masked)]
                if ndvi_masked.size >= num_of_pixels:
                # ndvi_masked = np.random.choice(ndvi_masked, num_of_pixels, replace=False)
                    ndvi_mean = np.nanmean(ndvi_masked) if ndvi_masked.size > 0 else None
                    ndvi_median = np.nanmedian(ndvi_masked) if ndvi_masked.size > 0 else None
                    break
                elif ndvi_masked.size > 0 and ndvi_masked.size < num_of_pixels:
                    # If there's some data but not enough, adjust the buffer and try again
                    buffer_dist *= 1.5
                else:
                    buffer_dist *= 1.5 
                    if buffer_dist > 500:  # Prevent infinite loop
                        ndvi_mean = None 
                        ndvi_median = None
                        break
            else:
                buffer_dist *= 1.5 
                if buffer_dist > 500:  # Prevent infinite loop
                    ndvi_mean = None 
                    ndvi_median = None
                    break
        
        return ndvi_mean, ndvi_median

    def preprocess_annotation_zones(self, buffer_distance=20):
        annotation_zones = [geom.buffer(buffer_distance) for geom in self.gdf.geometry]
        annotation_zones = [geom.buffer(0) for geom in self.gdf.geometry if not geom.is_valid]
        return annotation_zones

