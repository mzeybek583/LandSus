
## Stochastic Gradient Boosting 
## gradient boosting machine
#REmove everything
rm(list = ls())
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
#library(gbm)
# library(randomForest) 
#library(doParallel)

#registerDoParallel(cores = 3)
set.seed(917);        
# set the working directory to the data
#setwd("pathToDirHere")
# time 
time <- proc.time()
# import raw data tiffs
altitude <- "../Data/R_SVM/altitude.tif"
aspect <- "../Data/R_SVM/aspect.tif"
corine <- "../Data/R_SVM/corine.tif"
curvature <- "../Data/R_SVM/curvature.tif"
drenaj <- "../Data/R_SVM/dist_drenaj.tif"
fay <- "../Data/R_SVM/dist_fay.tif"
jeoloji <- "../Data/R_SVM/jeoloji.tif"
slope <- "../Data/R_SVM/slope.tif"
yol <- "../Data/R_SVM/dist_yol.tif"
twi <- "../Data/R_SVM/twi.tif"

# Landslide info
cls <- "../Data/study_area_heyelan/study_area_heyelan.tif"


# create list of files to make raster stack
#rasterlist1 <-  list.files('RGB', full.names=TRUE)

#rasterlist2 <-  list.files('RGB', full.names=TRUE, pattern="tif") 
getwd()
# create raster stack
raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)

# example syntax for stack from a list
#rstack1 <- stack(rasterlist1)

# check attributes
raster_data
proc.time()- time



names(raster_data)
data_df <- as.data.frame(raster_data, xy=TRUE)
rm(raster_data)
formula <- study_area_heyelan ~ altitude + aspect + corine + curvature + dist_drenaj + dist_fay + jeoloji +  slope + twi + dist_yol



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


## Random sampling to 10000 data
heyelan <- data_df_NA[data_df_NA$study_area_heyelan==1,]
heyelan_degil <- data_df_NA[data_df_NA$study_area_heyelan==0,]
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), 1000,replace=FALSE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), 1000,replace=FALSE),]

model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)
#split data into a train and test set 
train <- sample(nrow(model.data), 0.7*nrow(model.data), replace = FALSE)
TrainSet <- model.data[train,]
ValidSet <- model.data[-train,]
summary(TrainSet)
summary(ValidSet)
proc.time()- time

## All subsequent models are then run in parallel
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

set.seed(825)
model_gbm <- train(formula, data = TrainSet, 
                 method = "gbm", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE, preProc = c("center", "scale"))
proc.time()- time

model_gbm

## GBM Model

names(model_gbm)
model_gbm$results
summary(model_gbm)

gbmImp <- varImp(model_gbm, scale = TRUE)
gbmImp
saveRDS(gbmImp,"GBM_Train_varimportance")
trellis.par.set(caretTheme())
plot(model_gbm) 


#readRDS("GBM_Train_varimportance")

#png("TRAIN_varImportance_GBM.png")
tiff("TRAIN_varImportance_GBM.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_gbm, "super_model_GBM")
# Do not change

library(gmodels)
pred_train <-predict(model_gbm, TrainSet[,-13])
pred_valid <-predict(model_gbm, ValidSet[,-13])



library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$study_area_heyelan)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$study_area_heyelan)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"TRAIN_GBM_validation_ROC")
#aa <- readRDS("GBM_validation_ROC")
#png("TRAIN_roc_curve_train_GBM.png")
tiff("TRAIN_roc_curve_train_GBM.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (GBM)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "TRAIN_GBM.txt")

#png("TRAIN_roc_curve_valid_GBM.png")
tiff("TRAIN_roc_curve_valid_GBM.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (GBM)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()


perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "TRAIN_valid_GBM.txt")


## Apply to raster prediction
raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)
names(raster_data)
r1 <- raster::predict(raster_data, model_gbm, progress="text")
plot(r1)

writeRaster(r1,"TRAIN_GBM.tif", overwrite=TRUE)
proc.time() - time
