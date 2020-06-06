# Programmer: Dr. Mustafa ZEYBEK

# Logistic Regression
rm(list = ls()) #Remove everything
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
library(doParallel)
library(tidyverse)
library(nnet)

# N_CORES <- detectCores()
# cl <- makePSOCKcluster(N_CORES-1) # cores
# registerDoParallel(cl)

set.seed(917);   

# time 
time <- proc.time()

# DATA --------------------------------------------------------------------
# change
Tiff_path <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/data_v2/" 
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

library(e1071) 
library(rpart)

#Control NA
sapply(data_df, function(x)sum(is.na(x)))

# Delete NA rows
data_df_NA <- na.omit(data_df)

## Check NA 
sapply(data_df_NA, function(x)sum(is.na(x)))
# data_df_NA <- as.data.frame(lapply(data_df_NA, normalize))
min(data_df_NA$study_area_heyelan)

#Convert data to levels (factors)
col_names <- names(data_df_NA[,c(-1,-2)])
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
str(TrainSet)

# Create Model ------------------------------------------------------------

#model <- nnet::multinom(formula = formula, data = TrainSet)
model_cv <- caret::train(formula, data=TrainSet, method='multinom')
model_cv$resample
proc.time() -time
summary(model_cv)

#png("TRAIN_varImportance_LogReg.png")
tiff("RESULT/Model_varimportance_train_LogReg_v2.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()
# 
# predicted.classes <- model %>% predict(ValidSet[,-12])
# head(predicted.classes)
# mean(predicted.classes == ValidSet$study_area_heyelan)

names(model_cv)

setwd(file.path(Working_path))
gbmImp <- varImp(model_cv, scale = TRUE)
gbmImp

saveRDS(gbmImp,"RESULT/Model_varimportance_train_LogReg_v2")

# Do not change
saveRDS(model_cv, "RESULT/super_model_LogReg_v2")
# Do not change
library(gmodels)
pred_train <-predict(model_cv, TrainSet[,-12])
pred_valid <-predict(model_cv, ValidSet[,-12])

# Confusion Table Create
confusionMatrix(pred_train, TrainSet[,12], positive="1")
confusionMatrix(pred_valid, ValidSet[,12], positive="1")


# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_cv, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_LogReg_v2.tif", overwrite=TRUE)

#stopCluster(cl)

t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
