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
# Load necessary packages into Python
from subprocess import Popen
from getpass import getpass
from netrc import netrc
import argparse
import time
from datetime import datetime
import os
import requests
from requests.auth import HTTPBasicAuth
import geopandas as gpd
from getpass import getpass


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
            # 'L4A': 'GEDI04_A.002'
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

    def search(self, product, out_file=None):
        """
        Get GEDI granules

        :param product:
        :param out_file:
        :return:
        """
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
            granules = [c['links'][0]['href'] for c in cmr_response]
            granules = [x for x in granules if x.endswith('.h5')]
            if not out_file:
                return granules
            else:
                # Set up output text file name using the current datetime
                # outName = f"{product.replace('.', '_')}_GranuleList_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

                # Open file and write each granule link on a new line
                with open(out_file, "w") as gf:
                    for g in granules:
                        gf.write(f"{g}\n")
                # print(
                #     f"File containing links to intersecting {product} Version 2 data has been saved to:\n {os.getcwd()}\{outName}")

        except:
            # If the request did not complete successfully, print out the response from CMR
            print(
                requests.get(f"{self.base_url}{self.concept_ids[product]}&bounding_box={self.bbox.replace(' ', '')}&pageNum={page}").json())

    def download(self, product, username: str, password=None):
        """
        Download GEDI data

        :param product: Type of GEDI product [L1B, L2A, L2B, L4A]
        :param username: Username for NASA Earthdata
        :param password: Password for NASA Earthdata. Leave empty for hidden input.
        :return:
        """

        if not password:
            password = getpass()

        granules = self.search(product)

        for item in granules:
            url = item.split('.h5')[0] + '.h5'
            filename = url[url.rfind('/') + 1:]
            try:
                # submit the request using the session
                print('Downloading', url)
                session = SessionWithHeaderRedirection(username, password)
                response = session.get(url, stream=True)
                if response.status_code != 200:
                    print('Return Code:', response.status_code)
                # raise an exception in case of http errors
                response.raise_for_status()

                # save the file
                with open(filename, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        fd.write(chunk)

            except requests.exceptions.HTTPError as e:
                # handle any errors here
                print(e)
            break

        return


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
