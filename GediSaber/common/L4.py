# Common methods for GEDI L4 data
import os
from glob import glob

import geopandas as gpd
import h5py


def agbd_gdf_from_multi_h5(h5_dir):
    """
    Create GeoDataFrame from all h5 files in a directory. GeoDataFrame will consist
    of point data.

    :param h5_dir: Path of directory containing h5 files
    :return:
    """
    # Create dataframe
    lat_l = []
    lon_l = []
    agbd = []
    agbd_se = []
    glob_items = glob(os.path.join(h5_dir, '*.h5'))
    if not glob_items:
        raise ValueError(f'No .h5 files found in "{h5_dir}"')
    for subfile in glob_items:
        hf_in = h5py.File(subfile, 'r')
        for v in list(hf_in.keys()):
            if v.startswith('BEAM'):
                beam = hf_in[v]
                lat_l.extend(beam['lat_lowestmode'][:].tolist())
                lon_l.extend(beam['lon_lowestmode'][:].tolist())
                agbd.extend(beam['agbd'][:].tolist())
                agbd_se.extend(beam['agbd_se'][:].tolist())
        hf_in.close()
    geo_arr = list(zip(agbd, agbd_se, lat_l, lon_l))
    df = gpd.GeoDataFrame(geo_arr, columns=["agbd", "agbd_se", "lat_lowestmode", "lon_lowestmode"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_lowestmode, df.lat_lowestmode))
    return gdf
