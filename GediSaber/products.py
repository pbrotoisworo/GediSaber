import os
import warnings

import geopandas as gpd
import geoviews as gv
import h5py
import holoviews as hv
import numpy as np
from geoviews import tile_sources as gvts

from .common.l2 import (
    load_beam_data, load_metadata, gedi_orbit, create_gv_points
)
from .common.utils import point_visual, save_as_html

gv.extension('bokeh', 'matplotlib')


class L2B:

    def __init__(self, h5_file, verbose=True):
        """
        Class for handling GEDI L2B data.
        Code from: https://lpdaac.usgs.gov/resources/e-learning/getting-started-gedi-l2b-data-python

        :param h5_file: Path of H5 file
        :param verbose: Show log statements
        """

        self.data = h5py.File(h5_file, 'r')  # Read file using h5py
        self._h5_file = h5_file
        self._verbose = verbose

        # Load metadata
        self.beams = load_beam_data(self.data)
        self.metadata = load_metadata(self.data)

        # Get data objects
        self._data_objs = []
        self.data.visit(self._data_objs.append)  # Retrieve list of datasets
        self._gediSDS = [o for o in self._data_objs if isinstance(self.data[o], h5py.Dataset)]
        self._beamSDS = None

    def spatial_visualization(self, aoi, aoi_simplify_tolerance=0.001, add_aoi_to_webmap=False, fig_titles=None):

        default_fig_titles = {
            'geo_CanopyHeight': 'GEDI Canopy Height Map',
            'geo_Elevation': 'GEDI Elevation Height Map',
            'geo_PlantAreaIndex': 'GEDI Plant Area Index Map'
        }

        if not fig_titles:
            fig_titles = default_fig_titles

        beamNames = [g for g in self.data.keys() if g.startswith('BEAM')]
        # Set up lists to store data
        shotNum, dem, zElevation, zHigh, zLat, zLon, canopyHeight, quality, degrade, sensitivity, pai, beamI = ([] for i
                                                                                                                in
                                                                                                                range(
                                                                                                                    12))

        # Loop through each beam and open the SDS needed
        if self._verbose:
            print('Loading all beam data...')
        for b in beamNames:
            shotNum = self._load_sds_all_beams(shotNum, b, '/shot_number')
            dem = self._load_sds_all_beams(dem, b, '/digital_elevation_model')
            zElevation = self._load_sds_all_beams(zElevation, b, '/elev_lowestmode')
            zHigh = self._load_sds_all_beams(zHigh, b, '/elev_highestreturn')
            zLat = self._load_sds_all_beams(zLat, b, '/lat_lowestmode')
            zLon = self._load_sds_all_beams(zLon, b, '/lon_lowestmode')
            canopyHeight = self._load_sds_all_beams(canopyHeight, b, '/rh100')
            quality = self._load_sds_all_beams(quality, b, '/l2b_quality_flag')
            degrade = self._load_sds_all_beams(degrade, b, '/degrade_flag')
            sensitivity = self._load_sds_all_beams(sensitivity, b, '/sensitivity')
            [beamI.append(h) for h in [b] * len(self.data[[g for g in self._gediSDS if g.endswith('/shot_number') and b in g][0]][()])]
            [pai.append(h) for h in self.data[f'{b}/pai'][()]]

        # Convert lists to Pandas dataframe
        allDF = gpd.GeoDataFrame(
            {'Shot Number': shotNum, 'Beam': beamI, 'Latitude': zLat, 'Longitude': zLon, 'Tandem-X DEM': dem,
             'Elevation (m)': zElevation, 'Canopy Elevation (m)': zHigh, 'Canopy Height (rh100)': canopyHeight,
             'Quality Flag': quality, 'Plant Area Index': pai, 'Degrade Flag': degrade, 'Sensitivity': sensitivity})

        if aoi:
            aoiDF = gpd.GeoDataFrame.from_file(aoi)
            allDF['geometry'] = gpd.points_from_xy(allDF['Longitude'], allDF['Latitude'], crs='EPSG:4326')
            allDF = gpd.clip(allDF, aoiDF).drop(['Longitude', 'Latitude'], axis=1)
            if aoi_simplify_tolerance:
                aoiDF['geometry'] = aoiDF['geometry'].simplify(aoi_simplify_tolerance).buffer(0)  # Simplify geometry for webmap
            webmap_aoi = gv.Polygons(aoiDF['geometry']).opts(line_color='red',
                                                             fill_color='red',
                                                             fill_alpha=0.4,
                                                             hover_fill_alpha=0.2,
                                                             height=700,
                                                             width=1000)

        allDF['Shot Number'] = allDF['Shot Number'].astype(str)  # Convert shot number to string

        vdims = []
        for f in allDF:
            if f not in ['geometry']:
                vdims.append(f)

        allDF['Canopy Height (rh100)'] = allDF['Canopy Height (rh100)'] / 100  # Convert canopy height from cm to m

        if self._verbose:
            print('Generating geo webmaps...')

        # Set basemap which will be used for all maps
        basemap = gvts.EsriImagery

        title = fig_titles.get('geo_CanopyHeight') if fig_titles.get('geo_CanopyHeight') else \
            default_fig_titles.get('geo_CanopyHeight')
        points = create_gv_points('canopy_height', allDF, vdims, title)
        webmap = basemap * points
        if aoi and add_aoi_to_webmap:
            webmap = webmap_aoi * webmap
        save_as_html(webmap, 'geo_CanopyHeight')

        title = fig_titles.get('geo_Elevation') if fig_titles.get('geo_Elevation') else \
            default_fig_titles.get('geo_Elevation')
        points = create_gv_points('elevation', allDF, vdims, title)
        webmap = basemap * points
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        save_as_html(webmap, 'geo_Elevation')

        title = fig_titles.get('geo_PlantAreaIndex') if fig_titles.get('geo_PlantAreaIndex') else \
            default_fig_titles.get('geo_PlantAreaIndex')
        points = create_gv_points('pai', allDF, vdims, title)
        webmap = basemap * points
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        save_as_html(webmap, 'geo_PlantAreaIndex')

    def generate_transects(self, beam_id, aoi):

        # Open all of the desired SDS
        self._beamSDS = [g for g in self._gediSDS if beam_id in g]  # Subset to single beam
        canopyHeight = self._load_sds('/rh100')
        pavd = self.data[f'{beam_id}/pavd_z'][()]
        shotNums = self.data[f'{beam_id}/shot_number'][()]

        # Create a shot index
        shotIndex = np.arange(shotNums.size)

        # Convert RH100 from cm to m
        canopyHeight = canopyHeight / 100

        # Set up an empty list to append to
        pavdA = []
        for i in range(len(pavd)):

            # If any of the values are fill value, set to nan
            pavdF = [np.nan]
            for p in range(len(pavd[i])):
                if pavd[i][p] != -9999:
                    pavdF.append(pavd[i][p])  # If the value is not fill value, append to list
            pavdA.append(pavdF)  # Append back to master list

        # Take the DEM, GEDI-produced Elevation, and Canopy height and add to a Pandas dataframe
        transectDF = gpd.GeoDataFrame(
            {'Shot Index': shotIndex,
             'Shot Number': shotNums,
             'Latitude': self._load_sds('/lat_lowestmode'),
             'Longitude': self._load_sds('/lon_lowestmode'),
             'Tandem-X DEM': self._load_sds('/digital_elevation_model'),
             'Elevation (m)': self._load_sds('/elev_lowestmode'),
             'Canopy Elevation (m)': self._load_sds('/elev_highestreturn'),
             'Canopy Height (rh100)': canopyHeight,
             'Quality Flag': self._load_sds('/l2b_quality_flag'),
             'Degrade Flag': self._load_sds('/degrade_flag'),
             'Plant Area Volume Density': pavdA,
             'Sensitivity': self._load_sds('/sensitivity')
             })
        transectDF = transectDF.where(transectDF['Quality Flag'].ne(0))  # Set any poor quality returns to NaN
        transectDF = transectDF.where(transectDF['Degrade Flag'].ne(1))
        transectDF = transectDF.where(transectDF['Sensitivity'] > 0.95)
        transectDF = transectDF.dropna()  # Drop all of the rows (shots) that did not pass the quality filtering above

        if aoi:
            aoiDF = gpd.GeoDataFrame.from_file(aoi)
            transectDF['geometry'] = gpd.points_from_xy(
                x=transectDF['Longitude'],
                y=transectDF['Latitude'],
                crs='EPSG:4326'
            )
            transectDF = gpd.clip(transectDF, aoiDF).drop(['geometry'], axis=1)
            if self._verbose:
                print('Clipped dataframe size:', len(transectDF))

        # Plot Canopy Height
        canopyVis = hv.Scatter((transectDF['Shot Index'], transectDF['Canopy Height (rh100)']))
        canopyVis.opts(color='darkgreen', height=500, width=900, title=f'GEDI L2B Full Transect {beam_id}',
                       fontsize={'title': 16, 'xlabel': 16, 'ylabel': 16}, size=0.1, xlabel='Shot Index',
                       ylabel='Canopy Height (m)')
        save_as_html(canopyVis, 'transect_CanopyHeightAlongIndex')

        # Plot absolute heights
        # Plot Digital Elevation Model
        demVis = hv.Scatter((transectDF['Shot Index'], transectDF['Tandem-X DEM']), label='Tandem-X DEM')
        demVis = demVis.opts(color='black', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5)
        # Plot GEDI-Retrieved Elevation
        zVis = hv.Scatter((transectDF['Shot Index'], transectDF['Elevation (m)']), label='GEDI-derived Elevation')
        zVis = zVis.opts(color='saddlebrown', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5)
        # Plot Canopy Top Elevation
        rhVis = hv.Scatter((transectDF['Shot Index'], transectDF['Canopy Elevation (m)']), label='Canopy Top Elevation')
        rhVis = rhVis.opts(color='darkgreen', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5,
                           tools=['hover'], xlabel='Shot Index', ylabel='Elevation (m)')
        # Combine all three scatterplots
        file_basename = os.path.basename(self._h5_file).split('.')[0]
        webmap = (demVis * zVis * rhVis).opts(show_legend=True, legend_position='top_left',
                                     fontsize={'title': 15, 'xlabel': 16, 'ylabel': 16},
                                     title=f'{beam_id} Full Transect: {file_basename}')
        save_as_html(webmap, 'transect_ElevationAlongIndex')

        if self._verbose:
            print('Plotting profile transects')

        # Calculate along-track distance
        distance = np.arange(0.0, len(transectDF.index) * 60, 60)  # GEDI Shots are spaced 60 m apart
        transectDF['Distance'] = distance  # Add Distance as a new column in the dataframe

        pavdAll = []
        dz = self.data[f'{beam_id}/ancillary/dz'][0]  # Get vertical step size
        for j, s in enumerate(transectDF.index):
            pavdShot = transectDF['Plant Area Volume Density'][s]
            elevShot = transectDF['Elevation (m)'][s]
            pavdElev = []

            # Remove fill values
            if np.isnan(pavdShot).all():
                continue
            else:
                del pavdShot[0]
            for i, e in enumerate(range(len(pavdShot))):
                if pavdShot[i] > 0:
                    pavdElev.append(
                        (distance[j], elevShot + dz * i,
                         pavdShot[i]))  # Append tuple of distance, elevation, and PAVD
            pavdAll.append(pavdElev)  # Append to final list

        canopyElevation = [p[-1][1] for p in
                           pavdAll]  # Grab the canopy elevation by selecting the last value in each PAVD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path1 = hv.Path(pavdAll, vdims='PAVD').options(color='PAVD', clim=(0, 0.3), cmap='Greens', line_width=8,
                                                           colorbar=True,
                                                           width=950, height=500, clabel='PAVD',
                                                           xlabel='Distance Along Transect (m)',
                                                           ylabel='Elevation (m)',
                                                           fontsize={'title': 16, 'xlabel': 16, 'ylabel': 16,
                                                                     'xticks': 12, 'yticks': 12,
                                                                     'clabel': 12, 'cticks': 10})

        path2 = hv.Curve((distance, transectDF['Elevation (m)']), label='Ground Elevation').options(color='black',
                                                                                                    line_width=2)
        path3 = hv.Curve((distance, canopyElevation), label='Canopy Top Elevation').options(color='grey',
                                                                                            line_width=1.5)

        # Plot all three together
        path = path1 * path2 * path3
        path.opts(height=500, width=980, ylim=(min(transectDF['Elevation (m)']) - 5, max(canopyElevation) + 5),
                  xlabel='Distance Along Transect (m)', ylabel='Elevation (m)', legend_position='bottom_right',
                  fontsize={'title': 15, 'xlabel': 15, 'ylabel': 15, 'xticks': 14, 'yticks': 14, 'legend': 14},
                  title=f'GEDI L2B {beam_id}')
        save_as_html(path, 'transect_ElevationAlongDistance')

    def visualize_orbit(self, out_html_path=None, overlay=None, beam_id=None):
        """
        Visualize the path of the GEDI data. Output is HTML file.

        :param beam_id: Which ID to use for visualization. If None, it will use 'BEAM0000'.
        """

        if not beam_id:
            beam_id = 'BEAM0000'

        latslons = gedi_orbit(self.data, beam_id)

        # Take the lat/lon dataframe and convert each lat/lon to a shapely point
        latslons['geometry'] = gpd.points_from_xy(latslons['Longitude'], latslons['Latitude'], crs='EPSG:4326')

        # # Convert to a Geodataframe
        latslons = latslons.drop(columns=['Latitude', 'Longitude'])

        # Create a list of geodataframe columns to be included as attributes in the output map
        vdims = []
        for f in latslons:
            if f not in ['geometry']:
                vdims.append(f)

        # Call the function for plotting the GEDI points
        out_html_path = out_html_path if out_html_path else 'GEDI_orbit'
        if overlay:
            aoi = gpd.GeoDataFrame.from_file(overlay)  # Import geojson as GeoDataFrame
            webmap = gv.Polygons(aoi['geometry']).opts(line_color='red', color=None) * point_visual(latslons, vdims=vdims)
            save_as_html(webmap, out_html_path)
        else:
            save_as_html(point_visual(latslons, vdims), out_html_path)

    def _load_sds(self, target):
        return self.data[[g for g in self._beamSDS if g.endswith(target)][0]][()]

    def _load_sds_all_beams(self, lst, val, target):
        [lst.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith(target) and val in g][0]][()]]
        return lst
