
## Stochastic Gradient Boosting 
## gradient boosting machine
#REmove everything
rm(list = ls())
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
#library(doParallel)

#registerDoParallel(cores = 3)
set.seed(917);   

#setwd("pathToDirHere") # set the working directory to the data

# time 
time <- proc.time()


# DATA --------------------------------------------------------------------
# import raw data tiffs
altitude <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/altitude.tif"
aspect <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/aspect.tif"
corine <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/corine.tif"
curvature <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/curvature.tif"
drenaj <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/dist_drenaj.tif"
fay <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/dist_fay.tif"
jeoloji <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/jeoloji.tif"
slope <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/slope.tif"
yol <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/dist_yol.tif"
twi <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/twi.tif" 
#twi <- "../Data/R_SVM/twi.tif"

# Landslide Ground Truth DATA

cls <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/study_area_heyelan/study_area_heyelan.tif"

# Save Exports to RESULT folder --------------------------------------------------
mainDir <- "/home/mzeybek/LandSus/Code/"
subDir <- "RESULT"

dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
setwd(file.path(mainDir))

# create raster stack
raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)

# check attributes
raster_data
proc.time()- time
names(raster_data)
data_df <- as.data.frame(raster_data, xy=TRUE)
rm(raster_data)

# Train Formula -----------------------------------------------------------

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

## Random sampling to sampling data
# Sample variable
smpl <- 500
rto <- 0.7 # Train vs Test raio

## Random sampling data
heyelan <- data_df_NA[data_df_NA$study_area_heyelan==1,]
heyelan_degil <- data_df_NA[data_df_NA$study_area_heyelan==0,]
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), smpl, replace=FALSE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), smpl, replace=FALSE),]

model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)
#split data into a train and test set 
train <- sample(nrow(model.data), rto*nrow(model.data), replace = FALSE)
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

# Create Model ------------------------------------------------------------

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

#gbmImp <- varImp(model_gbm, scale = TRUE)
#gbmImp
#saveRDS(gbmImp,"RESULT/GBM_Train_varimportance")
#trellis.par.set(caretTheme())
#plot(model_gbm) 

#readRDS("GBM_Train_varimportance")

#png("TRAIN_varImportance_GBM.png")
#tiff("RESULT/TRAIN_varImportance_GBM.tiff", units="cm", width=8, height=8, res=600)
#plot(gbmImp, top = 10)
#dev.off()

# Do not change
saveRDS(model_gbm, "RESULT/super_model_GBM")
# Do not change

library(gmodels)
pred_train <-predict(model_gbm, TrainSet[,-13])
pred_valid <-predict(model_gbm, ValidSet[,-13])

# ROC plots ---------------------------------------------------------------

library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$study_area_heyelan)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$study_area_heyelan)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"RESULT/TRAIN_GBM_validation_ROC")
#aa <- readRDS("GBM_validation_ROC")
#png("TRAIN_roc_curve_train_GBM.png")
tiff("RESULT/TRAIN_roc_curve_train_GBM.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (GBM)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "RESULT/TRAIN_GBM.txt")

#png("TRAIN_roc_curve_valid_GBM.png")
tiff("RESULT/TRAIN_roc_curve_valid_GBM.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (GBM)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()


perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "RESULT/TRAIN_valid_GBM.txt")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)
names(raster_data)
r1 <- raster::predict(raster_data, model_gbm, progress="text")
plot(r1)

writeRaster(r1,"RESULT/TRAIN_GBM.tif", overwrite=TRUE)
proc.time() - time
