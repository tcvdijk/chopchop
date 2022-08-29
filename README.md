# chopchop

ChopChop - A Lightweight Pipeline for Geographic object Extraction using Tile Classification

*Thomas C. van Dijk, Martina Flörke, Nina Grundmann, Thorben Uschan*

## Example run

0. Acquire digital orthophotos in GeoTIFF format and make ground truth for the points of interest visible on it. 

1. Chop any large digital orthophoto GeoTIFF files into 150x150 pixel tiles stored in the directory called ```groundtruth```.
We tell it that we have a point of interest (POI) collection called ```poi.geojson``` in the same coordinate reference system.
Chop will put the tiles that contain a POI into a subdirectory called ```pos``` and those that do not into a subdirectory called ```neg```.
```
mkdir groundtruth
mkdir groundtruth/pos
mkdir groundtruth/neg
python chop.py dop.tif groundtruth --poi poi.geojson
```

2. Make sure the positive and negative classes are balanced for more successful learning.
```
python balance.py groundtruth
```

3. Fine-tune a pretrained classification neural net. See the paper for details.
```
python learn.py groundtruth my-model --epochs 10 --finetuning 10
```

4. Chop a GeoTIFF for which ground truth is not available and then use the model to classify
```
mkdir test
mkdir test/yes
mkdir test/no
python chop.py other-dop.tif test
python classify.py my-model test
```

5. The recognized POI tiles are now in ```test/yes```. 