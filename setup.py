from setuptools import setup, find_packages

setup(
    name="satellite_datacube",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'rasterio',
        'matplotlib',
        'simplejson', 
        'h5py',
        'fiona',
        'pandas',
    ],
    author="Paul Höhn",
    author_email="paul.hoen@outlook.de",
    description="A package to handle satellite image time series with data cubes.",
    license="MIT",
    keywords="satellite datacube",
    url="https://github.com/PaulH97/SatelliteDataCube.git",
)
