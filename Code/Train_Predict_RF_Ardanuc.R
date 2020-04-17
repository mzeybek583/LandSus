# Programmer: Dr. Mustafa ZEYBEK

## Random Forest

rm(list = ls()) #Remove everything
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 

#library(doParallel)
#registerDoParallel(cores = 3)

set.seed(917);   
time <- proc.time()# time 

# DATA --------------------------------------------------------------------
# change
Tiff_path <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/R_SVM/" 
Working_path <- "/home/mzeybek/LandSus/Code/" # change
smpl <- 500 # Sample variable
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

formula <- study_area_heyelan ~ altitude + aspect + corine + 
  curvature + dist_drenaj + dist_fay + jeoloji +  slope + twi + dist_yol

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

## Random sampling data
heyelan <- data_df_NA[data_df_NA$study_area_heyelan==1,]
heyelan_degil <- data_df_NA[data_df_NA$study_area_heyelan==0,]
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), smpl, replace=FALSE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), 
                                                        smpl, replace=FALSE),]

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
## RF Model
model_rf <-train(formula, data= TrainSet, method='rf',
                 trControl = fitControl, metric="RMSE",
                 preProcess = c("center","scale"), 
             #    ntree = 1000,
                 nodesize=100)

#                 metric='ROC', 
#tuneGrid = data.frame(mtry = 3), 
#                 trControl=tr)

#model_rf2 <-train(formula, data= TrainSet, method='RRF') 
                 #                 metric='ROC', 
                 #tuneGrid = data.frame(mtry = 3), 
 #                trControl=tr)
proc.time()- time
plot(model_rf)
model_rf
names(model_rf)
model_rf$results
summary(model_rf)

# Variable Importance
gbmImp <- varImp(model_rf, scale = TRUE)
gbmImp

setwd(file.path(Working_path))

saveRDS(gbmImp,file = "RESULT/Model_varimportance_train_RF")
#readRDS("SVM_Train_varimportance")

#png("TRAIN_varImportance_SVM.png")
tiff("RESULT/Model_varimportance_train_RF.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Save Super Model --------------------------------------------------------

# Do not change
saveRDS(model_rf, "RESULT/super_model_RF")
# Do not change

library(gmodels)
pred_train <-predict(model_rf, TrainSet[,-12])
pred_valid <-predict(model_rf, ValidSet[,-12])

# ROC plots ---------------------------------------------------------------

library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$study_area_heyelan)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$study_area_heyelan)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"RESULT/ROC_Curve_valid_RF")
#aa <- readRDS("Logreg_validation_ROC")
#png("TRAIN_roc_curve_train_LogReg.png")
tiff("RESULT/ROC_Curve_train_RF.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (RF)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "RESULT/Perf_AUC_train_RF.txt")

#png("TRAIN_roc_curve_valid_SVM.png")
tiff("RESULT/ROC_Curve_valid_RF.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (RF)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "RESULT/Perf_AUC_RF.txt")
  
# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_rf, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_RF.tif", overwrite=TRUE)
t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
  