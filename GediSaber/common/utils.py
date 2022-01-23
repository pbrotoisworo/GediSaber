# Common tools for GediSaber
import geoviews as gv
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
