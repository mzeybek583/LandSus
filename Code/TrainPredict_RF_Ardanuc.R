
# Random Forest for Landslide Susceptibility
#REmove everything
rm(list = ls())
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
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
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), 5000,replace=FALSE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), 5000,replace=FALSE),]

model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)
#split data into a train and test set 
train <- sample(nrow(model.data), 0.7*nrow(model.data), replace = FALSE)
TrainSet <- model.data[train,]
ValidSet <- model.data[-train,]
summary(TrainSet)
summary(ValidSet)
proc.time()- time

## All subsequent models are then run in parallel
#train_control <- trainControl(method="LOOCV")
#ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
#model_glm <- glm(formula, data = TrainSet, family=binomial, trControl=train_control)
#model_glm <- train(formula, data = TrainSet, method = "bayesglm", family=binomial, trControl=train_control)
#model_glm <- train(formula, data = TrainSet, 
#                    trControl = trainControl(method = "cv", number = 5),
#                    method = "glm",
#                   family = "binomial") 
# library(kernlab)
#   #Model building 
#   svm.model <- ksvm(formula, data = TrainSet, prob.model=TRUE)
#   
#   #svm.model <-svm(formula, data = TrainSet, cost =100, gamma =1)
#   svm.model
#ctrl <- trainControl(method="repeatedcv",   # 10fold cross validation
#                     repeats=5,		    # do 5 repititions of cv
#summaryFunction=twoClassSummary,	# Use AUC to pick the best model
#                     classProbs=TRUE)
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
#grid <- expand.grid(sigma = c(.01, .015, 0.2),
#                    C = c(0.75, 0.9, 1, 1.1, 1.25)
#)

## RF Model

model_rf <-train(formula, data= TrainSet, family = identity, trControl = ctrl, tuneLength =5)

proc.time()- time

model_rf

names(model_rf)
model_rf$results
summary(model_rf)

gbmImp <- varImp(model_rf, scale = TRUE)
gbmImp
saveRDS(gbmImp,"RF_Train_varimportance")
#readRDS("SVM_Train_varimportance")

#png("TRAIN_varImportance_SVM.png")
tiff("TRAIN_varImportance_RF.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()


# Save Super Model --------------------------------------------------------


# Do not change
saveRDS(model_rf, "super_model_RF")
# Do not change

library(gmodels)
pred_train <-predict(model_rf, TrainSet[,-13])
pred_valid <-predict(model_rf, ValidSet[,-13])



# ROC plots ---------------------------------------------------------------

library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$study_area_heyelan)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$study_area_heyelan)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"TRAIN_RF_validation_ROC")
#aa <- readRDS("Logreg_validation_ROC")
#png("TRAIN_roc_curve_train_LogReg.png")
tiff("TRAIN_roc_curve_train_RF.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (RF)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "TRAIN_RF.txt")

#png("TRAIN_roc_curve_valid_SVM.png")
tiff("TRAIN_roc_curve_valid_RF.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (RF)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()


perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "TRAIN_valid_RF.txt")


# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)
names(raster_data)
r1 <- raster::predict(raster_data, model_rf, progress="text")
plot(r1)

writeRaster(r1,"TRAIN_RF.tif", overwrite=TRUE)
proc.time() - time
