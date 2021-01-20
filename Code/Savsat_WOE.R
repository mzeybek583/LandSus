
# Weight of Evidence
# WOE describes the relationship between a predictive variable 
# and a binary target variable.
# IV measures the strength of that relationship.

#References

#Good, I. (1950): Probability and the Weighting of Evidences. Charles Griffin, 
#London.

#Kullback, S. (1959): Information Theory and Statistics. Wiley, New York.

# http://127.0.0.1:28979/library/Information/doc/Information-vignette.html
#[1] Hastie, Trevor, Tibshirani, Robert and Friedman, Jerome. (1986), Elements 
#of Statistical Learning, Second Edition, Springer, 2009.

#[2] Kullback S., Information Theory and Statistics, John Wiley and Sons, 1959.

#[3] Shannon, C.E., A Mathematical Theory of Communication, Bell System 
#Technical Journal, 1948.

#[4] Shannon, CE. and Weaver, W. The Mathematical Theory of Communication. 
#Univ of Illinois Press, 1949.

#[5] GAM: the Predictive Modeling Silver Bullet, (via)
##

timestamp <- Sys.time()
rm(list = ls()) #Remove everything

#install.packages("Information")
library(Information)

# load the required packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
library(plyr)
library(recipes)
library(dplyr)
library(doParallel)
library(klaR)
library(parallel)
library(doSNOW)

# Calculate the number of cores
numberofcores = detectCores() -1  # review what number of cores does for your environment

cl <- makeCluster(numberofcores, type = "SOCK")

set.seed(917);   

# time
time <- proc.time() 

# RASTER DATA --------------------------(Change here)------------------------------------------

Tiff_path <- "D:/data"
Working_path <- "D:/Landsus/LandSus/Code" 
smpl <- 50000 # number of landslide pixels
rto <- 0.7 # Training and validating ratio


# -------------------------------------------------------------------------



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

formula <- landslide_non_landslide ~ Altitude + Aspect + Curvature + Distance_to_drainage + 
  Distance_to_faults + Distance_to_roads + Land_Cover + Lithology +  Slope + Slope_Length + TWI

#Control NA
sapply(data_df, function(x)sum(is.na(x)))

# Delete NA rows
data_df_NA <- na.omit(data_df)

## Check NA 
sapply(data_df_NA, function(x)sum(is.na(x)))

## Random sampling 
heyelan <- data_df_NA[data_df_NA$landslide_non_landslide==1,]
heyelan_degil <- data_df_NA[data_df_NA$landslide_non_landslide==0,]
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), smpl, replace=TRUE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), smpl, replace=TRUE),]

model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)

#split data into a train and test set 
train <- sample(nrow(model.data), rto*nrow(model.data), replace = TRUE)
TrainSet <- model.data[train,]

TrainSet <- lapply(TrainSet, as.numeric)
#TrainSet <- lapply(TrainSet, factor)
#TrainSet$landslide_non_landslide <- as.numeric(TrainSet$landslide_non_landslide)

TrainSet <- data.frame(TrainSet)
TrainSet <- TrainSet[-c(1:2)]

ValidSet <- model.data[-train,]
ValidSet <- lapply(ValidSet, as.numeric)
ValidSet <- data.frame(ValidSet)
ValidSet <- ValidSet[-c(1:2)]
summary(TrainSet)
summary(ValidSet)
proc.time()- time

## 10-fold cross-validation

# fitControl <- trainControl(method = "repeatedcv",
#                            number = 10,
#                            repeats = 10
#                            #classProbs = TRUE,
#                            #search = "random"
# )

# Create Model ------------------------------------------------------------

# With Cross Validation
IV <- create_infotables(data=TrainSet, y="landslide_non_landslide", 
                        valid=ValidSet, 
                        parallel=TRUE,
                        bins=5)

# Without CV
#IV <- create_infotables(data=TrainSet, y="landslide_non_landslide", parallel=TRUE, bins=5)


#Changing the Number of Bins The default number of bins is 10 but we can 
#choose a different number if we desire more granularity. Note that the IV 
#formula is fairly invariant to the number of bins. Also, note that the bins 
#are selected such that the bins are evenly sized, to the extent that it is
#possible (depending on the number of ties in the data).

knitr::kable(head(IV$Summary))

print(head(IV), row.names=FALSE)
print(IV$Tables, row.names=FALSE)

names <- names(IV$Tables)
plots <- list()
for (i in 1:length(names)){
  plots[[i]] <- plot_infotables(IV, names[i])
}
# Showing the top 18 variables
plots[1:11]

stopCluster(cl) 

sprintf("Processing time is %3.3f sec.", (proc.time()- time)[3])
