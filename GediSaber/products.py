import os
import warnings

import h5py
import numpy as np
import geopandas as gpd
import geoviews as gv
from geoviews import opts, tile_sources as gvts
import holoviews as hv
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
        self.beams = self._load_beam_data()
        self.metadata = self._load_metadata()

        # Get data objects
        self._data_objs = []
        self.data.visit(self._data_objs.append)  # Retrieve list of datasets
        self._gediSDS = [o for o in self._data_objs if isinstance(self.data[o], h5py.Dataset)]

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
            [shotNum.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/shot_number') and b in g][0]][()]]
            [dem.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/digital_elevation_model') and b in g][0]][()]]
            [zElevation.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/elev_lowestmode') and b in g][0]][()]]
            [zHigh.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/elev_highestreturn') and b in g][0]][()]]
            [zLat.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/lat_lowestmode') and b in g][0]][()]]
            [zLon.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/lon_lowestmode') and b in g][0]][()]]
            [canopyHeight.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/rh100') and b in g][0]][()]]
            [quality.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/l2b_quality_flag') and b in g][0]][()]]
            [degrade.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/degrade_flag') and b in g][0]][()]]
            [sensitivity.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith('/sensitivity') and b in g][0]][()]]
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

        title = fig_titles.get('geo_CanopyHeight') if fig_titles.get('geo_CanopyHeight') else \
            default_fig_titles.get('geo_CanopyHeight')
        webmap = (gvts.EsriImagery * gv.Points(allDF, vdims=vdims).options(color='Canopy Height (rh100)',cmap='plasma', size=3, tools=['hover'],
                                                          clim=(0,102), colorbar=True, clabel='Meters',
                                                          title=title,
                                                          fontsize={'xticks': 10, 'yticks': 10, 'xlabel':16, 'clabel':12,
                                                                    'cticks':10,'title':16,'ylabel':16})).options(height=500,
                                                                                                                  width=900)
        if aoi and add_aoi_to_webmap:
            webmap = webmap_aoi * webmap
        self._save_as_html(webmap, 'geo_CanopyHeight')

        title = fig_titles.get('geo_Elevation') if fig_titles.get('geo_Elevation') else \
            default_fig_titles.get('geo_Elevation')
        webmap = (gvts.EsriImagery * gv.Points(allDF, vdims=vdims).options(color='Elevation (m)', cmap='terrain', size=3,
                                                                  tools=['hover'],
                                                                  clim=(min(allDF['Elevation (m)']),
                                                                        max(allDF['Elevation (m)'])),
                                                                  colorbar=True, clabel='Meters',
                                                                  title=title,
                                                                  fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
                                                                            'clabel': 12,
                                                                            'cticks': 10, 'title': 16,
                                                                            'ylabel': 16})).options(height=500,
                                                                                                    width=900)
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        self._save_as_html(webmap, 'geo_Elevation')

        title = fig_titles.get('geo_PlantAreaIndex') if fig_titles.get('geo_PlantAreaIndex') else \
            default_fig_titles.get('geo_PlantAreaIndex')
        webmap = (gvts.EsriImagery * gv.Points(allDF, vdims=vdims).options(color='Plant Area Index', cmap='Greens', size=3,
                                                                  tools=['hover'],
                                                                  clim=(0, 1), colorbar=True, clabel='m2/m2',
                                                                  title=title,
                                                                  fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
                                                                            'clabel': 12,
                                                                            'cticks': 10, 'title': 16,
                                                                            'ylabel': 16})).options(height=500,
                                                                                                    width=900)
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        self._save_as_html(webmap, 'geo_PlantAreaIndex')

    def generate_transects(self, beam_id, aoi):

        # Open all of the desired SDS
        beamSDS = [g for g in self._gediSDS if beam_id in g]  # Subset to single beam

        dem = self.data[[g for g in beamSDS if g.endswith('/digital_elevation_model')][0]][()]
        zElevation = self.data[[g for g in beamSDS if g.endswith('/elev_lowestmode')][0]][()]
        zHigh = self.data[[g for g in beamSDS if g.endswith('/elev_highestreturn')][0]][()]
        zLat = self.data[[g for g in beamSDS if g.endswith('/lat_lowestmode')][0]][()]
        zLon = self.data[[g for g in beamSDS if g.endswith('/lon_lowestmode')][0]][()]
        canopyHeight = self.data[[g for g in beamSDS if g.endswith('/rh100')][0]][()]
        quality = self.data[[g for g in beamSDS if g.endswith('/l2b_quality_flag')][0]][()]
        degrade = self.data[[g for g in beamSDS if g.endswith('/degrade_flag')][0]][()]
        sensitivity = self.data[[g for g in beamSDS if g.endswith('/sensitivity')][0]][()]
        pavd = self.data[f'{beam_id}/pavd_z'][()]
        shotNums = self.data[f'{beam_id}/shot_number'][()]
        dz = self.data[f'{beam_id}/ancillary/dz'][0]  # Get vertical step size

        # Create a shot index
        shotIndex = np.arange(shotNums.size)

        canopyHeight = canopyHeight / 100  # Convert RH100 from cm to m

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
            {'Shot Index': shotIndex, 'Shot Number': shotNums, 'Latitude': zLat, 'Longitude': zLon,
             'Tandem-X DEM': dem, 'Elevation (m)': zElevation, 'Canopy Elevation (m)': zHigh,
             'Canopy Height (rh100)': canopyHeight, 'Quality Flag': quality, 'Degrade Flag': degrade,
             'Plant Area Volume Density': pavdA, 'Sensitivity': sensitivity})
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
        self._save_as_html(canopyVis, 'transect_CanopyHeightAlongIndex')

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
        self._save_as_html(webmap, 'transect_ElevationAlongIndex')

        if self._verbose:
            print('Plotting profile transects')

        # Calculate along-track distance
        distance = np.arange(0.0, len(transectDF.index) * 60, 60)  # GEDI Shots are spaced 60 m apart
        transectDF['Distance'] = distance  # Add Distance as a new column in the dataframe

        pavdAll = []
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
        self._save_as_html(path, 'transect_ElevationAlongDistance')

    def visualize_orbit(self, out_html_path=None, overlay=None, beam_id=None):
        """
        Visualize the path of the GEDI data. Output is HTML file.

        :param beam_id: Which ID to use for visualization. If None, it will use 'BEAM0000'.
        """

        if not beam_id:
            beam_id = 'BEAM0000'

        lonSample, latSample, shotSample, qualitySample, beamSample = [], [], [], [], []  # Set up lists to store data

        # Open the SDS
        lats = self.data[f'{beam_id}/geolocation/lat_lowestmode'][()]
        lons = self.data[f'{beam_id}/geolocation/lon_lowestmode'][()]
        shots = self.data[f'{beam_id}/geolocation/shot_number'][()]
        quality = self.data[f'{beam_id}/l2b_quality_flag'][()]

        # Take every 100th shot and append to list
        for i in range(len(shots)):
            if i % 100 == 0:
                shotSample.append(str(shots[i]))
                lonSample.append(lons[i])
                latSample.append(lats[i])
                qualitySample.append(quality[i])
                beamSample.append(beam_id)

        # Write all of the sample shots to a dataframe
        latslons = gpd.GeoDataFrame(
            {'Beam': beamSample, 'Shot Number': shotSample, 'Longitude': lonSample, 'Latitude': latSample,
             'Quality Flag': qualitySample})

        # Take the lat/lon dataframe and convert each lat/lon to a shapely point
        # latslons['geometry'] = latslons.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
        latslons['geometry'] = gpd.points_from_xy(latslons['Longitude'], latslons['Latitude'], crs='EPSG:4326')

        # # Convert to a Geodataframe
        # latslons = gpd.GeoDataFrame(latslons)
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
            webmap = gv.Polygons(aoi['geometry']).opts(line_color='red', color=None) * self._pointVisual(latslons, vdims=vdims)
            self._save_as_html(webmap, out_html_path)
        else:
            self._save_as_html(self._pointVisual(latslons, vdims), out_html_path)

    @staticmethod
    def _save_as_html(gv_obj, out_path):
        if out_path.endswith('.html'):
            out_path = out_path.rstrip('.html')
        render = gv.renderer('bokeh')
        render.save(gv_obj, out_path)

    @staticmethod
    def _pointVisual(features, vdims):
        """Function for visualizing GEDI points"""
        return (gvts.EsriImagery * gv.Points(features, vdims=vdims).options(tools=['hover'], height=500, width=900,
                                                                            size=5,
                                                                            color='yellow',
                                                                            fontsize={'xticks': 10, 'yticks': 10,
                                                                                      'xlabel': 16, 'ylabel': 16}))

    def _load_beam_data(self):
        beam_list = [g for g in self.data.keys() if g.startswith('BEAM')]
        beams = {beam:{} for beam in beam_list}
        for beam_id in beam_list:
            power = self.data[beam_id].attrs['description']
            beams[beam_id]['POWER'] = 'FULL' if 'Full' in power else 'COVERAGE'
        return beams

    def _load_metadata(self):
        metadata = {}
        h5_data = self.data['METADATA']['DatasetIdentification']
        for attribute in h5_data.attrs:
            metadata[attribute] = h5_data.attrs[attribute]

        return metadata
