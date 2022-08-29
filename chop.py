import argparse
parser = argparse.ArgumentParser(description="Chop GeoTIFF into tiles; optionally classify by whether the tile contains a point from a geojson point set (poi).")
parser.add_argument('geotiff', type=str,
                    help="Filename of GeoTIFF image to chop up.")
parser.add_argument('outputdir', type=str,
                    help="Directory where to put the individual tiles.")
parser.add_argument('-p','--poi', type=str,
                    help="Filename of GeoJSON containing point-of-interest set.")
parser.add_argument('-t','--tilesize', type=int, default=150,
                    help="Side length of the tiles in pixels. (Default: 150)")
parser.add_argument('-s','--stride', type=int, default=-1,
                    help="Step size from one tile to the next in pixels. (Default: -1 = same as tilesize) ")
parser.add_argument('--pos', type=str, default='pos',
                    help="Directory name for the tiles that contain a point. (Default: 'pos')")
parser.add_argument('--neg', type=str, default='neg',
                    help="Directory name for the tiles that do not contain a point. (Default: 'neg')")
args = vars(parser.parse_args())

### settings

input_image_filename = args['geotiff']
input_poi_filename = args['poi']

tile_side = args['tilesize']
tile_stride = tile_side
if args['stride']!=-1:
    tile_stride = args['stride']

if input_poi_filename:
    # split based on poi dataset
    output_dir_pos = args['outputdir'] + '/' + args['pos'] + '/'
    output_dir_neg = args['outputdir'] + '/' + args['neg'] + '/'
else:
    # just put everything in the same dir
    output_dir_pos = args['outputdir'] + '/'
    output_dir_neg = args['outputdir'] + '/'

### imports

from osgeo import gdal
import cv2 as cv
import numpy as np
import json
from os.path import isdir
if not isdir(output_dir_pos):
    print("Directory for positives does not exist: ", output_dir_pos)
    exit(1)
if not isdir(output_dir_neg):
    print("Directory for negatives does not exist: ", output_dir_neg)
    exit(2)
# colored terminal messages when supported
try:
    from termcolor import colored
except:
    def colored(a,b):
        return a 

### helper functions

def rect_contains_point( x1, x2, y1, y2, ps ):
    for p in ps:
        if x1<=p[0] and p[0]<=x2 and y1<=p[1] and p[1]<=y2:
            return True
    return False

### open geojson

if input_poi_filename:
    with open(input_poi_filename,"r") as f:
        pois_json = json.load(f)
    pois = [ poi["geometry"]["coordinates"] for poi in pois_json["features"] ]
else:
    # we have no pois; everything will be classified negative.
    pois = []

### open big image

ds = gdal.Open(input_image_filename)
print( "Driver:", ds.GetDriver().ShortName, ds.GetDriver().LongName)
x_numpixels = ds.RasterXSize
y_numpixels = ds.RasterYSize
print( "Size is", x_numpixels, " x ", y_numpixels, " x ", ds.RasterCount )

gt = ds.GetGeoTransform()
input_proj = ds.GetProjection()
x_start = gt[0]
x_pixelsize = gt[1]
y_start = gt[3]
y_pixelsize = gt[5]
print("start x", x_start, ", pixel = ", x_pixelsize)
print("start y", y_start, ", pixel = ", y_pixelsize)

rs = ds.GetRasterBand(1).ReadAsArray().astype(float)
gs = ds.GetRasterBand(2).ReadAsArray().astype(float)
bs = ds.GetRasterBand(3).ReadAsArray().astype(float)


# cut the big image into tiles;
# put them into two different directories based on whether
# they contain a poi or not.

gtiff_driver = gdal.GetDriverByName('GTiff')
num_pos = 0
num_neg = 0
for x in range(0, x_numpixels - tile_side, tile_stride):
    for y in range(0, y_numpixels - tile_side, tile_stride):
        # set up information
        tile_name = f"{input_image_filename}-tile{tile_side}-{x:06d}-{y:06d}"
        # where are we?
        rect_x_min = x_start + x_pixelsize * x
        rect_x_max = x_start + x_pixelsize * (x + tile_side)
        rect_x_min, rect_x_max = sorted((rect_x_min, rect_x_max))
        rect_y_min = y_start + y_pixelsize * y
        rect_y_max = y_start + y_pixelsize * (y + tile_side)
        rect_y_min, rect_y_max = sorted((rect_y_min, rect_y_max))

        # are we positive or negative?
        pos = rect_contains_point( rect_x_min, rect_x_max, rect_y_min, rect_y_max, pois )
        if pos:
            fname = output_dir_pos + tile_name
            print(colored("POS","green"), fname)
            num_pos += 1
        else:
            fname = output_dir_neg + tile_name
            print(colored("NEG","red"), fname)
            num_neg += 1

        # get the image data
        tile_rs = rs[ y : y+tile_side ,  x : x+tile_side ]
        tile_gs = gs[ y : y+tile_side ,  x : x+tile_side ]
        tile_bs = bs[ y : y+tile_side ,  x : x+tile_side ]
        #tile_img = np.dstack((tile_bs,tile_gs,tile_rs))
        #cv.imwrite(fname+'.jpg', tile_img,[cv.IMWRITE_JPEG_QUALITY,90])

        dataset = gtiff_driver.Create(fname+'.tif', tile_side, tile_side, 3, gdal.GDT_Byte)
        tile_gt = ( gt[0] + x*gt[1],
                    gt[1], gt[2],
                    gt[3] + y*gt[5],
                    gt[4], gt[5] )
        dataset.SetGeoTransform(tile_gt)
        dataset.SetProjection(input_proj)
        dataset.GetRasterBand(1).WriteArray(tile_rs)
        dataset.GetRasterBand(2).WriteArray(tile_gs)
        dataset.GetRasterBand(3).WriteArray(tile_bs)
        dataset.FlushCache()
        del dataset


print("done :)")
print("#pos =",num_pos)
print("#neg =",num_neg)