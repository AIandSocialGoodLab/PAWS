'''

Author: Jiaqi Liu (jiaqili2 at andrew cmu edu)

Usage:
  python convert_coords.py

Basics:
  Convert shape files and TIF files into same coordinates, like wgs84.
  New files would be created and saved to sub-folders.

'''

from geopandas import read_file
from shutil import rmtree
from os import listdir, makedirs
from os.path import isfile, join, exists, splitext
from rasterio import open as rio_open
from rasterio import band
from rasterio.warp import calculate_default_transform, reproject, Resampling


def convert_shp_to_wgs84(input, output):
  '''convert from input to output to wgs84 CRS'''
  print(f'convert| {input}\n\t->{output}')
  world = read_file(input)
  world = world.to_crs(epsg=4326)
  world.to_file(output)


def prepare_dir(convert_path):
  '''prepare dir, remove if exists'''
  if exists(convert_path):
    rmtree(convert_path)
  makedirs(convert_path)


def get_shp_files_in_dir(base_path):
  '''get shape files in dir'''
  shp_files = [f for f in listdir(base_path)
               if isfile(join(base_path, f)) and '.shp' == splitext(f)[-1]]
  return shp_files


def get_tif_files_in_dir(base_path):
  '''get tif files in dir'''
  tif_files = [f for f in listdir(base_path)
               if isfile(join(base_path, f)) and '.tif' == splitext(f)[-1]]
  return tif_files


def convert_files_in_dir(base_path, convert_path):
  '''convert all shape/tif files in dir, and save them to wgs84 dir'''
  shp_files = get_shp_files_in_dir(base_path)
  for f in shp_files:
    convert_shp_to_wgs84(join(base_path, f), join(convert_path, f))
  tif_files = get_tif_files_in_dir(base_path)
  for f in tif_files:
    convert_tif_to_wgs84(join(base_path, f), join(convert_path, f))


def convert_tif_to_wgs84(input, output):
  '''convert from input to output to wgs84 CRS'''
  print(f'convert| {input}\n\t->{output}')

  dst_crs = 'EPSG:4326'
  with rio_open(input) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio_open(output, 'w', **kwargs) as dst:
      for i in range(1, src.count + 1):
        reproject(
            source=band(src, i),
            destination=band(dst, i),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)


def convert(base_path):
  convert_path = join(base_path, 'wgs84')
  prepare_dir(convert_path)
  convert_files_in_dir(base_path, convert_path)


def main():
  base_path = '/Users/jiaqiliu/workspace/cmu/PAWS-workspace/Quick_Employment/QuickEmployment_Toy/Absolute_Coordinates/all_shp_inconsistent'
  convert(base_path)

if __name__ == '__main__':
  main()
