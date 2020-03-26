#### Landslide suscebility Mapping
##
#
#

# Load libraries

library(raster)
library(sp)
library(rgdal)

# set working directory to data folder
#setwd("pathToDirHere")
DEM <- raster("abc.tif")

# look at the raster attributes. 
DEM

plot(DEM, main="Digital Elevation Model, abc") # add title with main

col <- terrain.colors(5)
image(DEM, zlim=c(250,375), main="Digital Elevation Model (DEM)", col=col)



# Change legend appearance ------------------------------------------------


# First, expand right side of clipping rectangle to make room for the legend turn xpd off
#par(xpd = FALSE, mar=c(5.1, 4.1, 4.1, 4.5))

# Second, plot w/ no legend
#plot(DEM, col=col, breaks=brk, main="DEM with a Custom (but flipped) Legend", legend = FALSE)

# Third, turn xpd back on to force the legend to fit next to the plot.
#par(xpd = TRUE)

# Fourth, add a legend - & make it appear outside of the plot
#legend(par()$usr[2], 4110600,
#       legend = c("lowest", "a bit higher", "middle ground", "higher yet", "highest"), 
#       fill = col)






