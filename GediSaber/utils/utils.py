# Common tools for GediSaber
import os
from glob import glob

import geopandas as gpd
import geoviews as gv
import h5py
import numpy as np
from geoviews import tile_sources as gvts

from .sds import SdsDatasets


def point_visual(features, vdims):
    """Function for visualizing GEDI points"""
    basemap = gvts.EsriImagery
    points = gv.Points(
        features,
        vdims=vdims).options(tools=['hover'],
                             height=500,
                             width=900,
                             size=5,
                             color='yellow',
                             fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16, 'ylabel': 16})

    return basemap * points


def save_as_html(gv_obj, out_path):
    """
    Save geoviews object as HTML using boken renderer

    :param gv_obj:
    :param out_path:
    :return:
    """
    if out_path.endswith('.html'):
        out_path = out_path.rstrip('.html')
    render = gv.renderer('bokeh')
    render.save(gv_obj, out_path)


def sds_search_path(paths, beam, target_sds):
    results = [x for x in paths if x.endswith(target_sds) and beam in x]
    if results:
        return results[0]
    else:
        raise Exception(f'SDS "{target_sds}" not found.')


def gedi_orbit(h5_obj, paths, beam_id):
    """
    Load a representative GEDI orbit (every 100th shot) and store the orbit data in a GeoDataFrame for easy
    visualization.

    :param h5_obj: h5 file loaded using h5py package
    :param beam_id: Name of beam
    :return:
    """

    lon_sample = []
    lat_sample = []
    shot_sample = []
    quality_sample = []
    beam_sample = []

    # Load SDS paths
    lats_path = sds_search_path(paths, beam_id, 'lat_lowestmode')
    lons_path = sds_search_path(paths, beam_id, 'lon_lowestmode')
    shots_path = sds_search_path(paths, beam_id, 'shot_number')
    quality_path = sds_search_path(paths, beam_id, 'quality_flag')

    # Open the SDS
    lats = h5_obj[lats_path][()]
    lons = h5_obj[lons_path][()]
    shots = h5_obj[shots_path][()]
    quality = h5_obj[quality_path][()]

    # Take every 100th shot and append to list
    for i in range(len(shots)):
        if i % 100 == 0:
            shot_sample.append(str(shots[i]))
            lon_sample.append(lons[i])
            lat_sample.append(lats[i])
            quality_sample.append(quality[i])
            beam_sample.append(beam_id)

    # Write all of the sample shots to a dataframe
    latslons = gpd.GeoDataFrame(
        {'Beam': beam_sample,
         'Shot Number': shot_sample,
         'Longitude': lon_sample,
         'Latitude': lat_sample,
         'Quality Flag': quality_sample
         })

    return latslons


def gdf_from_multi_h5(h5_dir, column_and_paths):
    """


    :param h5_dir:
    :param paths: Dictionary where key is dataframe columns and value is h5 path
    :return:
    """

    # Make lists to store dataframe data
    data = {}
    for column_name in column_and_paths.keys():
        data[column_name] = []

    glob_items = glob(os.path.join(h5_dir, '*.h5'))
    if not glob_items:
        raise ValueError(f'No .h5 files found in "{h5_dir}"')

    # Loop through h5 files to create GDFs
    for subfile in glob_items:
        hf_in = h5py.File(subfile, 'r')
        sds = SdsDatasets(hf_in)
        # Load all paths in h5 data
        h5_paths = []
        hf_in.visit(h5_paths.append)

        # Loop through BEAM data
        for v in list(hf_in.keys()):
            if v.startswith('BEAM'):
                beam = hf_in[v]
                # Loop through user specified column names and h5 paths
                for column_name, h5_path in column_and_paths.items():
                    # PAV and PAI data have their own specific workflows
                    if h5_path.endswith('pai'):
                        [data[column_name].extend(h for h in hf_in[f'{v}/pai'][()])]
                    elif h5_path.endswith('pav_z'):
                        raise NotImplementedError('pav_z sds not implemented for this method yet')
                    else:
                        data[column_name].extend(beam[match_path_by_beam(h5_paths, v, h5_path)][:].tolist())

        hf_in.close()

    df_data = {}
    for column_name in data.keys():
        df_data[column_name] = data[column_name]

    # # Create dataframe
    df = gpd.GeoDataFrame(df_data)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))

    return gdf


def subset(h5_in, subset_geom, outfile, processing_level):
    """
    Subset GEDI h5 data

    :param h5_in: Input loaded h5 data
    :param subset_geom: Mask GeoDataFrame
    :param outfile: Path of output h5 file
    :param processing_level: Processing level of input GEDI data
    :return:
    """

    if isinstance(h5_in, str):
        h5_in = h5py.File(h5_in, 'r')

    hf_out = h5py.File(outfile, 'w')
    h5_paths = []
    h5_in.visit(h5_paths.append)

    # copy ANCILLARY and METADATA groups
    if processing_level == 'L4':
        var1 = ["/ANCILLARY", "/METADATA"]
    else:
        var1 = ["/METADATA"]
    for v in var1:
        h5_in.copy(h5_in[v], hf_out)

    # loop through BEAMXXXX groups
    for v in list(h5_in.keys()):
        if v.startswith('BEAM'):
            beam = h5_in[v]
            # find the shots that overlays the area of interest (GRSM)
            # if processing_level == 'L4':
            #     lat = beam['lat_lowestmode'][:]
            #     lon = beam['lon_lowestmode'][:]
            # else:
            #     lat = beam['geolocation/lat_lowestmode'][:]
            #     lon = beam['geolocation/lon_lowestmode'][:]
            lat = beam[match_path_by_beam(h5_paths, v, 'lat_lowestmode')][:]
            lon = beam[match_path_by_beam(h5_paths, v, 'lon_lowestmode')][:]
            i = np.arange(0, len(lat), 1)  # index
            geo_arr = list(zip(lat, lon, i))
            l4adf = gpd.GeoDataFrame(geo_arr, columns=["lat_lowestmode", "lon_lowestmode", "i"])
            l4agdf = gpd.GeoDataFrame(l4adf,
                                      geometry=gpd.points_from_xy(l4adf.lon_lowestmode, l4adf.lat_lowestmode))
            l4agdf.crs = "EPSG:4326"
            l4agdf_gsrm = l4agdf[l4agdf['geometry'].within(subset_geom.geometry[0])]
            indices = l4agdf_gsrm.i

            # copy BEAMS to the output file
            for key, value in beam.items():
                if isinstance(value, h5py.Group):
                    for key2, value2 in value.items():
                        group_path = value2.parent.name
                        # group_id = hf_out.require_group(group_path)
                        dataset_path = group_path + '/' + key2
                        if len(value2) == 1:
                            hf_out.create_dataset(dataset_path, data=value2[0])
                        else:
                            hf_out.create_dataset(dataset_path, data=value2[:][indices])
                        for attr in value2.attrs.keys():
                            hf_out[dataset_path].attrs[attr] = value2.attrs[attr]
                else:
                    group_path = value.parent.name
                    # group_id = hf_out.require_group(group_path)
                    dataset_path = group_path + '/' + key
                    hf_out.create_dataset(dataset_path, data=value[:][indices])
                    for attr in value.attrs.keys():
                        hf_out[dataset_path].attrs[attr] = value.attrs[attr]

    h5_in.close()
    hf_out.close()


def match_path_by_beam(lst, beam, target):
    out = [x for x in lst if (x.endswith(target)) and (beam in x)][0]
    out = out.lstrip(f'{beam}/')
    return out


if __name__ == '__main__':
    pass