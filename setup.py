from setuptools import setup, find_packages

setup(
    name='GediSaber',
    version=0.1,
    packages=find_packages(),
    url='https://github.com/pbrotoisworo/GediSaber',
    author='Panji Brotoisworo',
    description='Python package that automates GEDI LiDAR downloading and processing',
    install_requires=[
        'h5py',
        'shapely',
        'geopandas',
        'pandas',
        'geoviews',
        'holoviews',
        'contextily'
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License"
    ],
    keywords="LiDAR, GEDI, altimetry"
)
