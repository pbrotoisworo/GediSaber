# Common GEDI level 2 methods
import geopandas as gpd
import geoviews as gv


def create_gv_points(point_type, allDF, vdims, title):
    if point_type == 'elevation':
        points = gv.Points(allDF,
                           vdims=vdims).options(color='Elevation (m)',
                                                cmap='terrain',
                                                size=3,
                                                tools=['hover'],
                                                height=500,
                                                width=900,
                                                clim=(min(allDF['Elevation (m)']),
                                                      max(allDF['Elevation (m)'])),
                                                colorbar=True,
                                                clabel='Meters',
                                                title=title,
                                                fontsize={'xticks': 10,
                                                          'yticks': 10,
                                                          'xlabel': 16,
                                                          'clabel': 12,
                                                          'cticks': 10,
                                                          'title': 16,
                                                          'ylabel': 16})

    elif point_type == 'pai':
        points = gv.Points(allDF,
                           vdims=vdims).options(color='Plant Area Index',
                                                cmap='Greens',
                                                size=3,
                                                height=500,
                                                width=900,
                                                tools=['hover'],
                                                clim=(0, 1),
                                                colorbar=True,
                                                clabel='m2/m2',
                                                title=title,
                                                fontsize={'xticks': 10,
                                                          'yticks': 10,
                                                          'xlabel': 16,
                                                          'clabel': 12,
                                                          'cticks': 10,
                                                          'title': 16,
                                                          'ylabel': 16})
    elif point_type == 'canopy_height':
        points = gv.Points(allDF,
                           vdims=vdims).options(color='Canopy Height (rh100)',
                                                cmap='plasma',
                                                size=3,
                                                tools=['hover'],
                                                clim=(0, 102),
                                                colorbar=True,
                                                height=500,
                                                width=900,
                                                clabel='Meters',
                                                title=title,
                                                fontsize={'xticks': 10,
                                                          'yticks': 10,
                                                          'xlabel': 16,
                                                          'clabel': 12,
                                                          'cticks': 10,
                                                          'title': 16,
                                                          'ylabel': 16})
    else:
        raise ValueError(f'Unknown point_type argument: "{point_type}"')

    return points

def load_beam_data(h5_obj):
    """
    Load beam information and information on beam strength

    :param h5_obj: h5 file loaded using h5py package
    :return:
    """
    beam_list = [g for g in h5_obj.keys() if g.startswith('BEAM')]
    beams = {beam: {} for beam in beam_list}
    for beam_id in beam_list:
        power = h5_obj[beam_id].attrs['description']
        beams[beam_id]['POWER'] = 'FULL' if 'Full' in power else 'COVERAGE'
    return beams


def load_metadata(h5_obj):
    """
    Load general GEDI metadata

    :param h5_obj: h5 file loaded using h5py package
    :return:
    """
    metadata = {}
    h5_data = h5_obj['METADATA']['DatasetIdentification']
    for attribute in h5_data.attrs:
        metadata[attribute] = h5_data.attrs[attribute]

    return metadata


def gedi_orbit(h5_obj, beam_id):
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

    # Open the SDS
    lats = h5_obj[f'{beam_id}/geolocation/lat_lowestmode'][()]
    lons = h5_obj[f'{beam_id}/geolocation/lon_lowestmode'][()]
    shots = h5_obj[f'{beam_id}/geolocation/shot_number'][()]
    quality = h5_obj[f'{beam_id}/l2b_quality_flag'][()]

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

if __name__ == '__main__':
    pass
