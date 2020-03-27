## RSAGA example

## Load library

library(RSAGA)
env <- rsaga.env()

data(landslides) 
write.sgrd(data = dem, "dem", header = dem$header)

# List of SAGA libraries
rsaga.get.libraries()
rsaga.get.modules("ta_hydrology")

rsaga.get.usage(lib = "ta_hydrology", module = "SAGA Wetness Index")

params = list("dem.sgrd", "twi.sdat")
rsaga.geoprocessor(lib = "ta_hydrology", module = "SAGA Wetness Index", param = params)

rsaga.wetness.index("dem", "twi")

rsaga.hillshade("dem","hillshade")

library(raster) 

twi = raster::raster("twi.sdat") # shown is a version using tmap
#twi@file@nodatavalue[is.na(twi@file@nodatavalue)] <- 0
#rna <- reclassify(twi, cbind(NA, 0))
NAvalue(twi)
NAvalue(twi) <- 0

## Plot results
plot( twi, col = RColorBrewer::brewer.pal(n = 9, name = "Blues"))
hillshd <- raster::raster("hillshade.sdat")
NAvalue(hillshd)
NAvalue(hillshd) <- 0
plot(hillshd,col = gray.colors(20, start = 0, end = 1))
