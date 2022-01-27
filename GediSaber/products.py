import os
import warnings
from glob import glob

import contextily as ctx
import geopandas as gpd
import geoviews as gv
import h5py
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geoviews import tile_sources as gvts

from .utils.L2 import create_gv_points
from .utils.sds import SdsDatasets
from .utils.utils import point_visual, save_as_html, subset, gdf_from_multi_h5, gedi_orbit

gv.extension('bokeh', 'matplotlib')


class L2BMulti:

    def __init__(self, h5_dir, verbose=True):
        """
        Class to work with multiple L2B data

        :param h5_dir: Path containing L2B data
        :param verbose: Print statements
        """
        self._h5_dir = h5_dir
        self._verbose = verbose

    def subset(self, aoi, out_dir, overwrite_existing=True):
        h5_in = glob(os.path.join(self._h5_dir, '*.h5'))
        aoi = gpd.GeoDataFrame.from_file(aoi)
        for item in h5_in:
            outfile = os.path.join(out_dir, os.path.basename(item))
            if (os.path.exists(outfile) and overwrite_existing) or (not os.path.exists(outfile)):
                if self._verbose:
                    print('Generating subset for:', item)
                subset(item, aoi, outfile, 'L2')

    @staticmethod
    def run_analysis(aoi, input_dir, output_dir):
        """
        Create GeoDataFrame from all h5 files in a directory. GeoDataFrame will consist
        of point data.

        Included datasets for L2B are plant area index (PAI), canopy height (rh100),
        canopy elevation, Tandem-X DEM, ground elevation.

        :param aoi: Path of AOI for mask
        :param input_dir: Path of directory containing h5 files
        :param output_dir: Path of output files
        :return:
        """

        # Load sample data to get path structure
        sds = SdsDatasets(glob(os.path.join(input_dir, '*.h5'))[0])
        # Set dataframe columns (keys) and data (values)
        df_data = {
                'Tandem-X DEM': sds.search_path('geolocation/digital_elevation_model')[0][8:],
                'Elevation (m)': sds.search_path('geolocation/elev_lowestmode')[0][8:],
                'Canopy Elevation (m)': sds.search_path('geolocation/elev_highestreturn')[0][8:],
                'Canopy Height (m)': sds.search_path('rh100')[0][8:],
                'Plant Area Index': sds.search_path('pai')[0][8:],
                'Quality Flag': sds.search_path('l2b_quality_flag')[0][8:],
                'Longitude': sds.search_path('lon_lowestmode')[0][8:],
                'Latitude': sds.search_path('lat_lowestmode')[0][8:]
            }

        gdf = gdf_from_multi_h5(input_dir, df_data)

        # Add AOI to GDF
        aoi = gpd.GeoDataFrame.from_file(aoi)
        # Copy original DF column but add -9999 to indicate invalid data
        df_aoi_data = {k: -9999 for (k, v) in df_data.items()}
        df_aoi_data['Quality Flag'] = 0
        df_aoi_data['geometry'] = aoi.geometry.item()
        grsm_df = gpd.GeoDataFrame([df_aoi_data])
        # grsm_df = gpd.GeoDataFrame(
        #     [[-9999, -9999, -9999, -9999, 0, -9999, -9999, aoi.geometry.item()]],
        #     columns=list(df_data.keys()) + ['geometry'])

        gdf = pd.concat([gdf, grsm_df])
        gdf.crs = "EPSG:4326"
        # 3857 (Web Mercator) for map figure
        gdf_out = gdf.to_crs(epsg=3857)
        # Convert data in cm unit to meters
        gdf_out['Canopy Elevation (m)'] = gdf_out['Canopy Elevation (m)'] / 100
        # Plot AOI
        ax4 = gdf_out[-1:].plot(color='white', edgecolor='red', alpha=0.3, linewidth=5, figsize=(22, 7))

        # Canopy elevation
        gdf_out[gdf_out['Canopy Elevation (m)'] != -9999][:-1].plot(ax=ax4,
                                                                    column='Canopy Elevation (m)',
                                                                    alpha=0.1,
                                                                    linewidth=0,
                                                                    legend=True)
        ctx.add_basemap(ax4)
        output_fig = os.path.join(output_dir, 'canopy_elevation.png')
        plt.title('GEDI Canopy Elevation')
        plt.savefig(output_fig)

        # PAI
        ax4 = gdf_out[-1:].plot(color='white', edgecolor='red', alpha=0.3, linewidth=5, figsize=(22, 7))
        gdf_out[gdf_out['Quality Flag'] != 0][:-1].plot(ax=ax4, column='Plant Area Index', alpha=0.1, linewidth=0,
                                                        legend=True)
        ctx.add_basemap(ax4)
        output_fig = os.path.join(output_dir, 'pai.png')
        plt.title('Plant Area Index')
        plt.savefig(output_fig)

        # Create GeoJSON containing all attributes
        geojson_path = os.path.join(output_dir, 'L2B_combined.geojson')
        gdf_out = gdf_out[gdf_out['Quality Flag'] != 0][:-1]
        gdf_out = gdf_out.to_crs(epsg=4326)  # Switch back to 4326 for shapefiles
        gdf_out.drop(['Quality Flag'], axis=1).to_file(geojson_path, driver="GeoJSON")


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
        self.sds = SdsDatasets(self.data)
        self._h5_paths = self.sds.paths

        # Load metadata
        self.level = self.data['METADATA']['DatasetIdentification'].attrs['shortName'].split('_')[1]
        if self.level != 'L2B':
            raise ValueError(f'Input data is type {self.level}. Class accepts L2B only.')

        # Get data objects
        self._data_objs = []
        self.data.visit(self._data_objs.append)  # Retrieve list of datasets
        self._gediSDS = [o for o in self._data_objs if isinstance(self.data[o], h5py.Dataset)]
        self._beamSDS = None

    def subset(self, aoi, output_file, overwrite=False):

        aoi = gpd.GeoDataFrame.from_file(aoi)

        # Create a subset if it doesn't exist or overwrite is True
        if (os.path.exists(output_file) and overwrite is True) or \
                os.path.exists(output_file) is False:
            subset(self.data, aoi, output_file, 'L2')
        return output_file

    def spatial_visualization(self, aoi, aoi_simplify_tolerance=0.001, add_aoi_to_webmap=False, fig_titles=None):

        default_fig_titles = {
            'geo_CanopyHeight': 'GEDI Canopy Height Map',
            'geo_Elevation': 'GEDI Elevation Height Map',
            'geo_PlantAreaIndex': 'GEDI Plant Area Index Map'
        }

        if not fig_titles:
            fig_titles = default_fig_titles

        if self._verbose:
            print('Loading all beam data...')

        # Set up lists to store data
        shot_num = []
        dem = []
        z_elevation = []
        z_high = []
        z_lat = []
        z_lon = []
        canopy_height = []
        quality = []
        degrade = []
        sensitivity = []
        pai = []
        beam_i = []

        # Loop through all the target SDS and add to list
        beams = [g for g in self.data.keys() if g.startswith('BEAM')]

        [shot_num.extend(self.sds.elevation(beam)) for beam in beams]
        [dem.extend(self.sds.dem(beam)) for beam in beams]
        [z_elevation.extend(self.sds.elevation(beam)) for beam in beams]
        [z_high.extend(self.sds.canopy_elevation(beam)) for beam in beams]
        [z_lat.extend(self.sds.latitude(beam)) for beam in beams]
        [z_lon.extend(self.sds.longitude(beam)) for beam in beams]
        [canopy_height.extend(self.sds.canopy_height(beam)) for beam in beams]
        [quality.extend(self.sds.quality_flag(beam, self.level)) for beam in beams]
        [degrade.extend(self.sds.degrade_flag(beam)) for beam in beams]
        [sensitivity.extend(self.sds.sensitivity(beam)) for beam in beams]
        [pai.extend(self.sds.pai(beam)) for beam in beams]

        for b in beams:
            [beam_i.append(h) for h in
             [b] * len(self.data[[g for g in self._gediSDS if g.endswith('/shot_number') and b in g][0]][()])]

        # Convert lists to Pandas dataframe
        allDF = gpd.GeoDataFrame(
            {'Shot Number': shot_num,
             'Beam': beam_i,
             'Latitude': z_lat,
             'Longitude': z_lon,
             'Tandem-X DEM': dem,
             'Elevation (m)': z_elevation,
             'Canopy Elevation (m)': z_high,
             'Canopy Height (m)': canopy_height,
             'Quality Flag': quality,
             'Plant Area Index': pai,
             'Degrade Flag': degrade,
             'Sensitivity': sensitivity
             })

        if aoi:
            if self._verbose:
                print('Clipping data to AOI extent...')
            aoiDF = gpd.GeoDataFrame.from_file(aoi)
            allDF['geometry'] = gpd.points_from_xy(allDF['Longitude'], allDF['Latitude'], crs='EPSG:4326')
            allDF = gpd.clip(allDF, aoiDF).drop(['Longitude', 'Latitude'], axis=1)
            if aoi_simplify_tolerance:
                # Simplify geometry for faster webmap visualization
                aoiDF['geometry'] = aoiDF['geometry'].simplify(aoi_simplify_tolerance).buffer(0)
            webmap_aoi = gv.Polygons(aoiDF['geometry']).opts(line_color='red',
                                                             fill_color='red',
                                                             fill_alpha=0.4,
                                                             hover_fill_alpha=0.2,
                                                             height=700,
                                                             width=1000)
            aoi_df3857 = aoiDF.to_crs(epsg=3857)

        else:
            aoi_df3857 = None
            webmap_aoi = None

        all_df3857 = allDF.to_crs(epsg=3857)

        allDF['Shot Number'] = allDF['Shot Number'].astype(str)  # Convert shot number to string

        vdims = []
        for f in allDF:
            if f not in ['geometry']:
                vdims.append(f)

        # Convert canopy height from cm to m
        allDF['Canopy Height (m)'] = allDF['Canopy Height (m)'] / 100

        if self._verbose:
            print('Generating geo webmaps and figures...')

        # Set basemap which will be used for all maps
        basemap = gvts.EsriImagery

        title = fig_titles.get('geo_CanopyHeight') if fig_titles.get('geo_CanopyHeight') else \
            default_fig_titles.get('geo_CanopyHeight')
        points = create_gv_points('canopy_height', allDF, vdims, title)
        webmap = basemap * points
        # Save webmap
        if aoi and add_aoi_to_webmap:
            webmap = webmap_aoi * webmap
        save_as_html(webmap, 'geo_CanopyHeight')
        self._save_image(all_df3857, 'Canopy Height (m)', 'geo_CanopyHeight.png', 'GEDI Canopy Height', aoi_df3857)

        title = fig_titles.get('geo_Elevation') if fig_titles.get('geo_Elevation') else \
            default_fig_titles.get('geo_Elevation')
        points = create_gv_points('elevation', allDF, vdims, title)
        webmap = basemap * points
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        save_as_html(webmap, 'geo_Elevation')
        self._save_image(all_df3857, 'Canopy Elevation (m)', 'geo_Elevation.png', 'GEDI Canopy Elevation', aoi_df3857)

        title = fig_titles.get('geo_PlantAreaIndex') if fig_titles.get('geo_PlantAreaIndex') else \
            default_fig_titles.get('geo_PlantAreaIndex')
        points = create_gv_points('pai', allDF, vdims, title)
        webmap = basemap * points
        if aoi and add_aoi_to_webmap:
            webmap *= webmap_aoi
        save_as_html(webmap, 'geo_PlantAreaIndex')
        self._save_image(all_df3857, 'Plant Area Index', 'geo_PlantAreaIndex.png', 'GEDI Plant Area Index', aoi_df3857)

    def generate_transects(self, beam_id: str, aoi):

        # Open all of the desired SDS
        pavd = self.sds.pavd_z(beam_id)
        # Get canopy height (rh100) and convert from cm to m
        canopy_height = self.sds.canopy_height(beam_id) / 100
        shot_nums = self.sds.shot_number(beam_id)
        # Create a shot index
        shot_index = np.arange(shot_nums.size)

        # Set up an empty list to append to
        pavd_a = []
        for i in range(len(pavd)):

            # If any of the values are fill value, set to nan
            pavdF = [np.nan]
            for p in range(len(pavd[i])):
                if pavd[i][p] != -9999:
                    pavdF.append(pavd[i][p])  # If the value is not fill value, append to list
            pavd_a.append(pavdF)  # Append back to master list

        # Take the DEM, GEDI-produced Elevation, and Canopy height and add to a Pandas dataframe
        transect_df = gpd.GeoDataFrame(
            {'Shot Index': shot_index,
             'Shot Number': shot_nums,
             'Latitude': self.sds.latitude(beam_id),
             'Longitude': self.sds.longitude(beam_id),
             'Tandem-X DEM': self.sds.dem(beam_id),
             'Elevation (m)': self.sds.elevation(beam_id),
             'Canopy Elevation (m)': self.sds.canopy_elevation(beam_id),
             'Canopy Height (canopy_height)': canopy_height,
             'Quality Flag': self.sds.quality_flag(beam_id, self.level),
             'Degrade Flag': self.sds.degrade_flag(beam_id),
             'Plant Area Volume Density': pavd_a,
             'Sensitivity': self.sds.sensitivity(beam_id)
             })
        transect_df = transect_df.where(transect_df['Quality Flag'].ne(0))  # Set any poor quality returns to NaN
        transect_df = transect_df.where(transect_df['Degrade Flag'].ne(1))
        transect_df = transect_df.where(transect_df['Sensitivity'] > 0.95)
        transect_df = transect_df.dropna()  # Drop all of the rows (shots) that did not pass the quality filtering above

        if aoi:
            aoiDF = gpd.GeoDataFrame.from_file(aoi)
            transect_df['geometry'] = gpd.points_from_xy(
                x=transect_df['Longitude'],
                y=transect_df['Latitude'],
                crs='EPSG:4326'
            )
            transect_df = gpd.clip(transect_df, aoiDF).drop(['geometry'], axis=1)
            if self._verbose:
                print('Clipped dataframe size:', len(transect_df))

        # Plot Canopy Height
        canopyVis = hv.Scatter((transect_df['Shot Index'], transect_df['Canopy Height (canopy_height)']))
        canopyVis.opts(color='darkgreen', height=500, width=900, title=f'GEDI L2B Full Transect {beam_id}',
                       fontsize={'title': 16, 'xlabel': 16, 'ylabel': 16}, size=0.1, xlabel='Shot Index',
                       ylabel='Canopy Height (m)')
        save_as_html(canopyVis, 'transect_CanopyHeightAlongIndex')

        # Plot absolute heights
        # Plot Digital Elevation Model
        dem_vis = hv.Scatter((transect_df['Shot Index'], transect_df['Tandem-X DEM']), label='Tandem-X DEM')
        dem_vis = dem_vis.opts(color='black', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5)
        # Plot GEDI-Retrieved Elevation
        z_vis = hv.Scatter((transect_df['Shot Index'], transect_df['Elevation (m)']), label='GEDI-derived Elevation')
        z_vis = z_vis.opts(color='saddlebrown', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5)
        # Plot Canopy Top Elevation
        rh_vis = hv.Scatter((transect_df['Shot Index'], transect_df['Canopy Elevation (m)']),
                            label='Canopy Top Elevation')
        rh_vis = rh_vis.opts(color='darkgreen', height=500, width=900, fontsize={'xlabel': 16, 'ylabel': 16}, size=1.5,
                             tools=['hover'], xlabel='Shot Index', ylabel='Elevation (m)')
        # Combine all three scatterplots
        file_basename = os.path.basename(self._h5_file).split('.')[0]
        webmap = (dem_vis * z_vis * rh_vis).opts(show_legend=True, legend_position='top_left',
                                                 fontsize={'title': 15, 'xlabel': 16, 'ylabel': 16},
                                                 title=f'{beam_id} Full Transect: {file_basename}')
        save_as_html(webmap, 'transect_ElevationAlongIndex')

        if self._verbose:
            print('Plotting profile transects')

        # Calculate along-track distance
        distance = np.arange(0.0, len(transect_df.index) * 60, 60)  # GEDI Shots are spaced 60 m apart
        transect_df['Distance'] = distance  # Add Distance as a new column in the dataframe

        pavd_all = []
        dz = self.data[f'{beam_id}/ancillary/dz'][0]  # Get vertical step size
        for j, s in enumerate(transect_df.index):
            pavd_shot = transect_df['Plant Area Volume Density'][s]
            elev_shot = transect_df['Elevation (m)'][s]
            pavd_elev = []

            # Remove fill values
            if np.isnan(pavd_shot).all():
                continue
            else:
                del pavd_shot[0]
            for i, e in enumerate(range(len(pavd_shot))):
                if pavd_shot[i] > 0:
                    pavd_elev.append(
                        (distance[j], elev_shot + dz * i,
                         pavd_shot[i]))  # Append tuple of distance, elevation, and PAVD
            pavd_all.append(pavd_elev)  # Append to final list

        canopy_elevation = [p[-1][1] for p in
                           pavd_all]  # Grab the canopy elevation by selecting the last value in each PAVD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path1 = hv.Path(pavd_all, vdims='PAVD').options(color='PAVD', clim=(0, 0.3), cmap='Greens', line_width=8,
                                                            colorbar=True,
                                                            width=950, height=500, clabel='PAVD',
                                                            xlabel='Distance Along Transect (m)',
                                                            ylabel='Elevation (m)',
                                                            fontsize={'title': 16, 'xlabel': 16, 'ylabel': 16,
                                                                      'xticks': 12, 'yticks': 12,
                                                                      'clabel': 12, 'cticks': 10})

        path2 = hv.Curve((distance, transect_df['Elevation (m)']), label='Ground Elevation').options(color='black',
                                                                                                     line_width=2)
        path3 = hv.Curve((distance, canopy_elevation), label='Canopy Top Elevation').options(color='grey',
                                                                                             line_width=1.5)

        # Plot all three together
        path = path1 * path2 * path3
        path.opts(height=500, width=980, ylim=(min(transect_df['Elevation (m)']) - 5, max(canopy_elevation) + 5),
                  xlabel='Distance Along Transect (m)', ylabel='Elevation (m)', legend_position='bottom_right',
                  fontsize={'title': 15, 'xlabel': 15, 'ylabel': 15, 'xticks': 14, 'yticks': 14, 'legend': 14},
                  title=f'GEDI L2B {beam_id}')
        save_as_html(path, 'transect_ElevationAlongDistance')

    def visualize_orbit(self, out_html_path=None, overlay=None, beam_id=None):
        """
        Visualize the path of the GEDI data. Output is HTML file.

        :param beam_id: Which ID to use for visualization. If None, it will use 'BEAM0000'.
        :param out_html_path: Path to save HTML file containing webmap
        :param overlay: Optionally add polygon to GEDI orbit webmap
        """

        if not beam_id:
            beam_id = 'BEAM0000'

        latslons = gedi_orbit(self.data, self._h5_paths, beam_id)

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
        # Currently creating a orbit figure (.png) creates an image with weird dimensions
        # So it is just a webmap for now
        out_html_path = out_html_path if out_html_path else 'GEDI_orbit'
        if overlay:
            aoi = gpd.GeoDataFrame.from_file(overlay)  # Import geojson as GeoDataFrame
            webmap = gv.Polygons(aoi['geometry']).opts(line_color='red', color=None) * point_visual(latslons, vdims=vdims)
            save_as_html(webmap, out_html_path)
        else:
            save_as_html(point_visual(latslons, vdims), out_html_path)

    @staticmethod
    def _save_image(allDF, allDF_col, out_fig_path, fig_title=None, aoiDF=None, target_epsg_code=3857):
        if aoiDF is not None:
            crs_aoi = aoiDF.crs.srs
        crs_all = allDF.crs.srs

        # Check if GDFs are in target EPSG
        target_epsg_str = f'epsg:{target_epsg_code}'
        if aoiDF is not None:
            if crs_aoi != target_epsg_str:
                aoiDF = aoiDF.to_crs(target_epsg_code)
        if crs_all != target_epsg_str:
            allDF = allDF.to_crs(target_epsg_code)

        # Create figures
        if aoiDF is not None:
            ax = aoiDF.plot(color='white', edgecolor='red', alpha=0.3, linewidth=5, figsize=(22, 7))
            allDF.plot(ax=ax, column=allDF_col, alpha=0.1, linewidth=0, legend=True)
        else:
            ax = allDF.plot(column=allDF_col, alpha=0.1, linewidth=0, legend=True)
        ctx.add_basemap(ax)

        # Set up fig and save
        if fig_title:
            plt.title(fig_title)
        plt.savefig(out_fig_path)

    def _load_sds(self, target):
        return self.data[[g for g in self._beamSDS if g.endswith(target)][0]][()]

    def _load_sds_all_beams(self, lst, val, target):
        [lst.append(h) for h in self.data[[g for g in self._gediSDS if g.endswith(target) and val in g][0]][()]]
        return lst


class L4AMulti:

    def __init__(self, h5_dir, verbose=True):
        """
        Class for handling GEDI L4A data.
        Code from: https://github.com/ornldaac/gedi_tutorials/blob/main/2_gedi_l4a_subsets.ipynb

        :param h5_file: Path of H5 file
        :param verbose: Show log statements
        """
        self._verbose = verbose
        self._h5_dir = h5_dir

    def subset(self, aoi, out_dir, overwrite_existing=True):
        h5_in = glob(os.path.join(self._h5_dir, '*.h5'))
        aoi = gpd.GeoDataFrame.from_file(aoi)
        for item in h5_in:
            outfile = os.path.join(out_dir, os.path.basename(item))
            if (os.path.exists(outfile) and overwrite_existing) or (not os.path.exists(outfile)):
                if self._verbose:
                    print('Generating subset for:', item)
                subset(item, aoi, outfile, 'L4')

    @staticmethod
    def run_analysis(aoi, input_dir, output_dir):
        """
        Class method for extracting AGBD.

        :param aoi: AOI used to clip data
        :param input_dir: Path containing all h5 data to use for AGBD analysis
        :param output_dir: Directory to save output figures and GIS data
        :return:
        """

        # Load sample data to get path structure
        sds = SdsDatasets(glob(os.path.join(input_dir, '*.h5'))[0])
        df_data = {
            'AGBD': sds.search_path('agbd')[0][8:],
            'AGBD_SE': sds.search_path('agbd_se')[0][8:],
            'Quality Flag': sds.search_path('l4_quality_flag')[0][8:],
            'Latitude': sds.search_path('lat_lowestmode')[0][8:],
            'Longitude': sds.search_path('lon_lowestmode')[0][8:]
        }

        # Create GDF from all available h5 data
        gdf = gdf_from_multi_h5(input_dir, df_data)

        # Add AOI to GDF
        aoi = gpd.GeoDataFrame.from_file(aoi)
        # Copy original DF column but add -9999 to indicate invalid data
        df_aoi_data = {k: -9999 for (k, v) in df_data.items()}
        df_aoi_data['Quality Flag'] = 0
        df_aoi_data['geometry'] = aoi.geometry.item()
        grsm_df = gpd.GeoDataFrame([df_aoi_data])

        gdf = pd.concat([gdf, grsm_df])
        gdf.crs = "EPSG:4326"
        gdf_out = gdf.to_crs(epsg=3857)  # 3857 (Web Mercator) for map figure
        ax4 = gdf_out[-1:].plot(color='white', edgecolor='red', alpha=0.3, linewidth=5, figsize=(22, 7))
        gdf_out[gdf_out['AGBD'] != -9999][:-1].plot(ax=ax4, column='AGBD', alpha=0.1, linewidth=0,
                                                              legend=True)
        ctx.add_basemap(ax4)
        output_fig = os.path.join(output_dir, 'agbd.png')
        plt.title('Above Ground Biomass Density')
        plt.savefig(output_fig)

        # AGBD Standard Error of Prediction
        ax4 = gdf_out[-1:].plot(color='white', edgecolor='red', alpha=0.3, linewidth=5, figsize=(22, 7))
        gdf_out[gdf_out['AGBD_SE'] != -9999][:-1].plot(ax=ax4, column='AGBD_SE', alpha=0.1, linewidth=0,
                                                                 legend=True)
        ctx.add_basemap(ax4)
        output_fig = os.path.join(output_dir, 'agbd_se.png')
        plt.title('Above Ground Biomass Density Standard Error')
        plt.savefig(output_fig)

        geojson_path = os.path.join(output_dir, 'agbd.geojson')
        gdf_out = gdf_out[gdf_out['AGBD'] != -9999][:-1]
        gdf_out = gdf_out.to_crs(epsg=4326)  # Switch back to 4326 for shapefiles
        gdf_out.drop(['Quality Flag'], axis=1).to_file(geojson_path, driver="GeoJSON")
