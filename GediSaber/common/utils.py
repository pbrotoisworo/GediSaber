# Common tools for GediSaber

import geopandas as gpd
import geoviews as gv
import h5py
import numpy as np
from geoviews import tile_sources as gvts


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


def subset(h5_in, subset_geom, outfile):
    """
    Subset GEDI h5 data

    :param h5_in: Input loaded h5 data
    :param subset_geom: Mask GeoDataFrame
    :param outfile: Path of output h5 file
    :return:
    """

    hf_out = h5py.File(outfile, 'w')

    # copy ANCILLARY and METADATA groups
    var1 = ["/ANCILLARY", "/METADATA"]
    for v in var1:
        h5_in.copy(h5_in[v], hf_out)

    # loop through BEAMXXXX groups
    for v in list(h5_in.keys()):
        if v.startswith('BEAM'):
            beam = h5_in[v]
            # find the shots that overlays the area of interest (GRSM)
            lat = beam['lat_lowestmode'][:]
            lon = beam['lon_lowestmode'][:]
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


if __name__ == '__main__':
    pass