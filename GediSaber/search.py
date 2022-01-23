"""
---------------------------------------------------------------------------------------------------
 How to Access the LP DAAC Data Pool with Python
 The following Python code example demonstrates how to configure a connection to download data from
 an Earthdata Login enabled server, specifically the LP DAAC's Data Pool. Modified for GediSaber.
---------------------------------------------------------------------------------------------------
 Original Author: Cole Krehbiel
 Source: https://lpdaac.usgs.gov/resources/e-learning/spatial-querying-of-gedi-version-2-data-in-python/
---------------------------------------------------------------------------------------------------
"""
import os
import traceback
from getpass import getpass

import geopandas as gpd
import requests
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import orient


class GediFinder:

    def __init__(self, geom):

        self.base_url = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

        # Set up dictionary where key is GEDI shortname + version and value is CMR Concept ID
        # Formal product name
        self.concept_ids = {
            'GEDI01_B.002': 'C1908344278-LPDAAC_ECS',
            'GEDI02_A.002': 'C1908348134-LPDAAC_ECS',
            'GEDI02_B.002': 'C1908350066-LPDAAC_ECS'
        }
        # Shorthand name which is meant for users
        self.gedi_products = {
            'L1B': 'GEDI01_B.002',
            'L2A': 'GEDI02_A.002',
            'L2B': 'GEDI02_B.002',
        }
        self.geom = geom
        self.bbox = self._get_bbox()

    def _get_bbox(self):

        if isinstance(self.geom, str):
            df = gpd.read_file(self.geom)
        elif isinstance(self.geom, gpd.GeoDataFrame):
            df = self.geom
        else:
            raise Exception('Should be GeoDataFrame or path of geometry')

        return ','.join([str(x) for x in df.total_bounds])

    def search(self, product: str, date_range=None, out_file=None):
        """
        Get GEDI granules

        :param product: Type of GEDI to search for
        :param date_range: Filter for date range in format ('YYYY-MM-DD', 'YYYY-MM-DD')
        :param out_file: Granule URLs will be saved in a txt file
        :return:
        """

        if product == 'L4A':
            return self._search_L4A(date_range, out_file)
        else:
            product_str = product
            product = self.gedi_products[product]
        page = 1
        try:
            # Send GET request to CMR granule search endpoint w/ product concept ID,
            # bbox & page number, format return as json
            cmr_response = requests.get(
                f"{self.base_url}{self.concept_ids[product]}&bounding_box={self.bbox}&pageNum={page}").json()['feed']['entry']

            # If 2000 features are returned, move to the next page and submit another
            # request, and append to the response
            while len(cmr_response) % 2000 == 0:
                page += 1
                cmr_response += \
                requests.get(
                    f"{self.base_url}{self.concept_ids[product]}&bounding_box={self.bbox}&pageNum={page}").json()['feed']['entry']

            # CMR returns more info than just the Data Pool links, below use list
            # comprehension to return a list of DP links

            granule_sizes = []
            granule_urls = []
            granule_datetimes = []
            if cmr_response:
                for item in cmr_response:
                    # Get file size
                    granule_sizes.append(float(item['granule_size']))
                    # Get URL
                    granule_urls.append(item['links'][0]['href'])
                    # Get datetime
                    granule_datetimes.append(item['time_start'])

            df_granule = gpd.GeoDataFrame({
                'date': granule_datetimes,
                'url': granule_urls,
                'size': granule_sizes
            })

            # Filter datetime
            if date_range:
                df_granule = df_granule[(df_granule['date'] > date_range[0]) & (df_granule['date'] < date_range[1])]

            print(f"Total {product_str} granules found: ", len(df_granule))
            print("Total file size (MB): ", '{0:,.2f}'.format(df_granule['size'].sum()))

            if not out_file:
                return df_granule['url'].to_list()
            else:
                # Open file and write each granule link on a new line
                with open(out_file, "w") as gf:
                    for g in df_granule['url'].to_list():
                        gf.write(f"{g}\n")

        except Exception as e:
            # If the request did not complete successfully, print out the response from CMR
            # print(
            #     requests.get(f"{self.base_url}{self.concept_ids[product]}&bounding_box={self.bbox.replace(' ', '')}&pageNum={page}").json())
            traceback.print_exc()

    def download(self, product, date_range, output_folder, username: str, password=None, auto_confirm=True):
        """
        Download GEDI data

        :param product: Type of GEDI product [L1B, L2A, L2B, L4A]
        :param date_range: Filter for date range in format ('YYYY-MM-DD', 'YYYY-MM-DD')
        :param output_folder: Folder to save downloaded GEDI data
        :param username: Username for NASA Earthdata
        :param password: Password for NASA Earthdata. Leave empty for hidden input.
        :return:
        """

        if not password:
            password = getpass()

        if product == 'L4A':
            granules = self._search_L4A(date_range=date_range)
        else:
            granules = self.search(product, date_range)

        confirm = input('Proceed with download? [Y/n]')
        if auto_confirm is False:
            if confirm == 'Y':
                pass
            elif confirm == 'n':
                return
            else:
                print('Unknown input', f"{confirm}")
                return

        for i, item in enumerate(granules, start=1):
            url = item.split('.h5')[0] + '.h5'
            filename = url[url.rfind('/') + 1:]
            try:
                # submit the request using the session
                print(f'Downloading [{i}/{len(granules)}]:', url)
                session = SessionWithHeaderRedirection(username, password)
                response = session.get(url, stream=True)
                if response.status_code != 200:
                    print('Return Code:', response.status_code)
                # raise an exception in case of http errors
                response.raise_for_status()

                # save the file
                output_file = os.path.join(output_folder, filename)
                with open(output_file, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        fd.write(chunk)

            except requests.exceptions.HTTPError as e:
                # handle any errors here
                print(e)

        return

    def _search_L4A(self, date_range=None, out_file=None):
        # converting to WGS84 coordinate system
        grsm_poly = gpd.read_file(self.geom)
        grsm_epsg4326 = grsm_poly.to_crs(epsg=4326)

        # orienting coordinates clockwise
        grsm_epsg4326.geometry = grsm_epsg4326.geometry.apply(orient, args=(1,))

        # reducing number of vertices in the polygon
        # CMR has 1000000 bytes limit
        grsm_epsg4326 = grsm_epsg4326.simplify(0.0005)

        doi = '10.3334/ORNLDAAC/1986'  # GEDI L4A DOI

        # CMR API base url
        cmrurl = 'https://cmr.earthdata.nasa.gov/search/'

        doisearch = cmrurl + 'collections.json?doi=' + doi
        concept_id = requests.get(doisearch).json()['feed']['entry'][0]['id']

        geojson = {"shapefile": ("grsm.json", grsm_epsg4326.geometry.to_json(), "application/geo+json")}

        page_num = 1
        page_size = 2000  # CMR page size limit

        granule_arr = []

        while True:

            # defining parameters
            cmr_param = {
                "collection_concept_id": concept_id,
                "page_size": page_size,
                "page_num": page_num,
                "simplify-shapefile": 'true'  # this is needed to bypass 5000 coordinates limit of CMR
            }

            granulesearch = cmrurl + 'granules.json'
            response = requests.post(granulesearch, data=cmr_param, files=geojson)
            granules = response.json()['feed']['entry']

            if granules:
                for g in granules:
                    granule_url = ''
                    granule_poly = ''

                    # read file size
                    granule_size = float(g['granule_size'])
                    granule_datetime = g['time_start']

                    # reading bounding geometries
                    if 'polygons' in g:
                        polygons = g['polygons']
                        multipolygons = []
                        for poly in polygons:
                            i = iter(poly[0].split(" "))
                            ltln = list(map(" ".join, zip(i, i)))
                            multipolygons.append(
                                Polygon([[float(p.split(" ")[1]), float(p.split(" ")[0])] for p in ltln]))
                        granule_poly = MultiPolygon(multipolygons)

                    # Get URL of HDF5 files
                    for links in g['links']:
                        if 'title' in links and links['title'].startswith('Download') \
                                and links['title'].endswith('.h5'):
                            granule_url = links['href']
                    granule_arr.append([granule_datetime, granule_url, granule_size, granule_poly])

                page_num += 1
            else:
                break

        # adding bound as the last row into the dataframe
        # we will use this later in the plot
        granule_arr.append(['GRSM', 0, grsm_epsg4326.geometry.item()])

        # creating a pandas dataframe
        l4adf = gpd.GeoDataFrame(granule_arr, columns=["granule_date", "granule_url", "granule_size", "granule_poly"])

        # Drop granules with empty geometry
        l4adf = l4adf[l4adf['granule_poly'] != '']

        if date_range:
            l4adf = l4adf[(l4adf['granule_date'] > date_range[0]) & (l4adf['granule_date'] < date_range[1])]

        print("Total L4A granules found: ", len(l4adf.index) - 1)
        print("Total file size (MB): ", '{0:,.2f}'.format(l4adf['granule_size'].sum()))
        granule_list = [f'{x}\n' for x in l4adf['granule_url'] if '.h5' in x]
        if out_file:
            # Export to CSV
            with open(out_file, 'w') as f:
                f.writelines(granule_list)
        else:
            return granule_list


# overriding requests.Session.rebuild_auth to mantain headers when redirected
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):

        super().__init__()

        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):

        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:

            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                redirect_parsed.hostname != self.AUTH_HOST and \
                original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']

        return


if __name__ == '__main__':
    pass
