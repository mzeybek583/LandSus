# Programmer: Dr. Mustafa ZEYBEK
# Correlation test on landlside attiributes

## ONGOING !!

library(corrplot)

rm(list = ls()) #Remove everything
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
library(doParallel)
#The package klaR contains Naive Ba   yes classifier.
library(klaR)

#N_CORES <- detectCores()

#cl <- makePSOCKcluster(N_CORES-1) # cores
#registerDoParallel(cl)

set.seed(917);   

# time 
time <- proc.time()

# DATA --------------------------------------------------------------------
# change
Tiff_path <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/data_v3/" 
Working_path <- "/home/mzeybek/LandSus/Code/" # change
smpl <- 100 # Sample variable
rto <- 0.7 # Train vs Test raio

setwd(Tiff_path)

# Save Exports to RESULT folder
Result_Dir <- "RESULT" # Output file # Do not change
temp = list.files(path = Tiff_path , pattern="\\.tif$", 
                  full.names = FALSE, recursive = TRUE)
raster_data <- raster::stack(paste0(temp))
dir.create(file.path(Working_path, Result_Dir), showWarnings = FALSE)

# check attributes
raster_data
proc.time()- time
names(raster_data)
data_df <- as.data.frame(raster_data, xy=TRUE)

# Train Formula -----------------------------------------------------------

formula <- study_area_heyelan ~ altitude + aspect + corine + curvature + drainage + faults + geology +  slope + twi + roads


## Normalization function
# normalize <- function(x) { return((x - min(x)) / (max(x) - min(x))) }

#data_df_norm <- as.data.frame(lapply(data_df[], normalize))
library(e1071) 
library(rpart)

#Control NA
sapply(data_df, function(x)sum(is.na(x)))

# Delete NA rows
data_df_NA <- na.omit(data_df)

## Check NA 
sapply(data_df_NA, function(x)sum(is.na(x)))
#Convert data to levels (factors)
#col_names <- names(data_df_NA[,c(-1,-2)])
# do do it for some names in a vector named 'col_names'
#data_df_NA[col_names] <- lapply(data_df_NA[col_names] , numeric)
data_df_NA$study_area_heyelan <- as.factor(data_df_NA$study_area_heyelan)
str(data_df_NA)
## Random sampling to from each class at smpl count
model.data <- data.frame()
datalist = list()

for (i in 1:length(levels(data_df_NA$study_area_heyelan))) {
  out <- data_df_NA[data_df_NA$study_area_heyelan==i,]
  out1 <- out[sample(1:nrow(out), smpl, replace=FALSE),]
  out1$i <- i  # maybe you want to keep track of which iteration produced it?
  datalist[[i]] <- out1 # add it to your list
}
model.data = do.call(rbind, datalist)
#split data into a train and test set 
train <- sample(nrow(model.data), rto*nrow(model.data), replace = FALSE)
TrainSet <- model.data[train,]
ValidSet <- model.data[-train,]
summary(TrainSet)
summary(ValidSet)
proc.time()- time

# Plot Cor
C4 <- model.data[model.data$study_area_heyelan==4,]
#C4 <- C4[order(C4$x),]
C4 <- C4[,-c(1,2,12,14)]
C4 = apply(C4, 2, function(x) as.numeric(as.character(x)));

C5 <- model.data[model.data$study_area_heyelan==5,]
#C5 <- C5[order(C5$x),]
C5 <- C5[,-c(1,2,12,14)]
C5 = apply(C5, 2, function(x) as.numeric(as.character(x)));

str(C4)
cor_mat <- cor(C4, C5)

corrplot(cor_mat, method="circle")

summary(C4)
summary(C5)
library(psych)

an1 <- describe(C4)
an2 <- describe(C5)

pairs.panels(C4)
