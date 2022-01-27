from typing import Union

import h5py


class SdsDatasets:

    def __init__(self, h5_obj: Union[str, h5py.File]):
        """
        Class to handle data loading from GEDI h5 datasets

        :param h5_obj: Path of h5 file or h5 file loaded using h5py.File
        """
        if isinstance(h5_obj, str):
            self._h5 = h5py.File(h5_obj, 'r')
        elif isinstance(h5_obj, h5py.File):
            self._h5 = h5_obj
        else:
            raise NotImplementedError('Unsupported input for SdsLayers class.')

        self.paths = []
        self._h5.visit(self.paths.append)
        self.level = self._h5['METADATA']['DatasetIdentification'].attrs['shortName'].split('_')[1]
        self._GEDI_BEAMS = [
            'BEAM0000',
            'BEAM0001',
            'BEAM0010',
            'BEAM0011',
            'BEAM0101',
            'BEAM0110',
            'BEAM1000',
            'BEAM1011'
        ]

        # Load SDS
        self.sds_datasets = [x for x in self.paths if isinstance(self._h5[x], h5py.Dataset)]

    def _beam_sds(self, beam):
        """Filter SDS data by beam"""
        return [x for x in self.paths if beam in x]

    def dem(self, beam):
        """Get Tandem-X DEM data array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/digital_elevation_model')][0]][()]

    def elevation(self, beam):
        """Get ground elevation array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/elev_lowestmode')][0]][()]

    def latitude(self, beam):
        """Get latitude array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/lat_lowestmode')][0]][()]

    def longitude(self, beam):
        """Get longitude array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/lon_lowestmode')][0]][()]

    def canopy_height(self, beam):
        """Get canopy height (rh100) array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/rh100')][0]][()]

    def canopy_elevation(self, beam):
        """Get canopy elevation array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith('/elev_highestreturn')][0]][()]

    def shot_number(self, beam):
        """Get shot number array"""
        return self._h5[f'{beam}/shot_number'][()]

    def quality_flag(self, beam, processing):
        """Get quality flag array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith(f'/{processing.lower()}_quality_flag')][0]][()]

    def degrade_flag(self, beam):
        """Get degrade flag array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith(f'/degrade_flag')][0]][()]

    def pavd_z(self, beam):
        """Get pavd_z array"""
        return self._h5[f'{beam}/pavd_z'][()]

    def pai(self, beam):
        """Get pai array"""
        return self._h5[f'{beam}/pai'][()]

    def sensitivity(self, beam):
        """Get sensitivity array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith(f'/sensitivity')][0]][()]

    def agbd(self, beam):
        """Get above ground biomass density array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith(f'/agbd')][0]][()]

    def agbd_se(self, beam):
        """Get above ground biomass density array"""
        return self._h5[[g for g in self._beam_sds(beam) if g.endswith(f'/agbd_se')][0]][()]

    def load_dataset(self, target_sds, beam=None):
        """
        Load a dataset in the H5 file

        :param target_sds: Target SDS to load
        :param beam: Load data from a specific beam
        :return:
        """
        out_data = []

        # Load data for all beams
        if not beam:
            for beam in self._GEDI_BEAMS:
                out_data.extend(self._h5[self.search_path(target_sds, beam)][()])
        else:
            # Load data for one beam only
            out_data.extend(self._h5[self.search_path(target_sds, beam)][()])

        return out_data

    def search_path(self, target_sds, beam_or_group=None):
        """
        Search h5 data for specific SDS path

        :param target_sds: Name of SDS
        :param beam_or_group: Beam or H5 group used as an additional filter for string matching.
            In GEDI data this would be something like the BEAMXXXX name, ANCILLARY, or METADATA groups.
        :return:
        """
        # Search for path with a particular beam
        if beam_or_group:
            results = [x for x in self.paths if x.endswith(target_sds) and beam_or_group in x]
            if not results:
                raise Exception(f'No SDS found with name "{target_sds}" and beam "{beam_or_group}"')
            results = results[0]
        else:
            results = []
            for beam in self._GEDI_BEAMS:
                results_initial = [x for x in self.paths if x.endswith(target_sds) and beam in x]
                if not results_initial:
                    raise Exception(f'No SDS found with name "{target_sds}" and beam "{beam}"')
                results += results_initial
        return results


if __name__ == '__main__':
    pass
