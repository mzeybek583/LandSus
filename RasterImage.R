# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)

# set the working directory to the data
#setwd("pathToDirHere")

# import tiffs
band19 <- "band19.tif"
band34 <- "band34.tif"
band58 <- "band58.tif"


# create list of files to make raster stack
rasterlist1 <-  list.files('RGB', full.names=TRUE)


rasterlist2 <-  list.files('RGB', full.names=TRUE, pattern="tif") 
getwd()
# create raster stack
rgbRaster <- stack(band19,band34,band58)

# example syntax for stack from a list
#rstack1 <- stack(rasterlist1)

# check attributes
rgbRaster

## class       : RasterStack 
## dimensions  : 502, 477, 239454, 3  (nrow, ncol, ncell, nlayers)
## resolution  : 1, 1  (x, y)
## extent      : 256521, 256998, 4112069, 4112571  (xmin, xmax, ymin, ymax)
## coord. ref. : +proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0 
## names       : band19, band34, band58 
## min values  :     84,    116,    123 
## max values  :  13805,  15677,  14343

# plot stack
plot(rgbRaster)
plotRGB(rgbRaster,r=3,g=2,b=1, stretch = "lin")
# view histogram of reflectance values for all rasters
hist(rgbRaster)

# determine the desired extent
rgbCrop <- c(256770.7,256959,4112140,4112284)

# crop to desired extent
rgbRaster_crop <- crop(rgbRaster, rgbCrop)

# view cropped stack
plot(rgbRaster_crop)
plotRGB(rgbRaster_crop,r=3,g=2,b=1, stretch = "lin")
# create raster brick
rgbBrick <- brick(rgbRaster)

# check attributes
rgbBrick

# view object size
object.size(rgbBrick)

## 5759744 bytes

object.size(rgbRaster)

## 41592 bytes

# view raster brick
plotRGB(rgbBrick,r=3,g=2,b=1, stretch = "Lin")

# Make a new stack in the order we want the data in 
orderRGBstack <- stack(rgbRaster$band58,rgbRaster$band34,rgbRaster$band19)

# write the geotiff
# change overwrite=TRUE to FALSE if you want to make sure you don't overwrite your files!
writeRaster(orderRGBstack,"rgbRaster.tif","GTiff", overwrite=TRUE)


# import multi-band raster as stack
multiRasterS <- stack("rgbRaster.tif") 

# import multi-band raster direct to brick
multiRasterB <- brick("rgbRaster.tif") 

# view raster
plot(multiRasterB)
plotRGB(multiRasterB,r=1,g=2,b=3, stretch="lin")
