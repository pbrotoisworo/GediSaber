# Common GEDI level 2 methods
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
                           vdims=vdims).options(color='Canopy Height (canopy_height)',
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


if __name__ == '__main__':
    pass
