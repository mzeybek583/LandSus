## Classification and Regression Trees (CART) 

timestamp <- Sys.time()
rm(list = ls()) #Remove everything

# load the required packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
library(plyr)
library(recipes)
library(dplyr)
library(doParallel)
library(doSNOW)

# Calculate the number of cores
numberofcores = detectCores() -1  # review what number of cores does for your environment

cl <- makeCluster(numberofcores, type = "SOCK")
set.seed(917);   

# time
time <- proc.time() 

# RASTER DATA --------------------------------------------------------------------

Tiff_path <- "D:/data"
Working_path <- "D:/Landsus/LandSus/Code" 
smpl <- 1000 # number of landslide pixels
rto <- 0.7 # Training and validating ratio

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
TrainSet <- data.frame(lapply(TrainSet, as.numeric))   # shows all columns are numeric
TrainSet$landslide_non_landslide <- as.factor(TrainSet$landslide_non_landslide)

ValidSet <- model.data[-train,]
ValidSet <- data.frame(lapply(ValidSet, as.numeric))   # shows all columns are numeric
ValidSet$landslide_non_landslide <- as.factor(ValidSet$landslide_non_landslide)

summary(TrainSet)
summary(ValidSet)
proc.time()- time

## 10-fold cross-validation

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3)


# Create Model ------------------------------------------------------------

model_nb <-train(formula, data = TrainSet,
                   method ='nb',
                  trControl = fitControl, 
                 #metric = "RMSE",
                 # metric = "ROC",
                 #tuneLength = 10,
                  #preProcess = c("center","scale"), 
                  allowParallel = TRUE
                  )
#getModelInfo(model_nb)

proc.time()- time

plot(model_nb)
model_nb

names(model_nb)
model_nb$results
summary(model_nb)

# Variable Importance
gbmImp <- varImp(model_nb, scale = F)
gbmImp

setwd(file.path(Working_path))

saveRDS(gbmImp,file = "RESULT/Model_varimportance_train_NB")

tiff("RESULT/Model_varimportance_train_NB.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_nb, "RESULT/super_model_NB")
# Do not change

library(gmodels)
pred_train <- as.numeric(predict(model_nb, TrainSet[,-10]))
pred_valid <-as.numeric(predict(model_nb, ValidSet[,-10]))

# ROC plots ---------------------------------------------------------------

library(ROCR)

#nbperf = performance(pred_train, "tpr", "fpr")

pred_training <- prediction(predictions = pred_train, labels = TrainSet$landslide_non_landslide)
#nbperf = performance(pred_training, "tpr", "fpr")

pred_validating <- prediction(predictions = pred_valid, labels = ValidSet$landslide_non_landslide)

perf_training <- performance(pred_training, measure = "tpr", x.measure = "fpr")
perf_validating <- performance(pred_validating, measure = "tpr", x.measure = "fpr")

saveRDS(perf_training,"RESULT/ROC_Curve_training_NB")
saveRDS(perf_validating,"RESULT/ROC_Curve_validation_NB")

tiff("RESULT/ROC_Curve_training_NB.tiff", units="cm", width=8, height=8, res=600)
plot(perf_training, main = "Success rate for NB", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred_training, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)

## Export accuracy
dput(perf.auc, "RESULT/Perf_AUC_training_NB.txt")

tiff("RESULT/ROC_Curve_validation_NB.tiff", units="cm", width=8, height=8, res=600)
plot(perf_validating, main = "Prediction rate for NB", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc_valid <- performance(pred_validating, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)

## Export accuracy
dput(perf.auc_valid, "RESULT/Perf_AUC_prediction_NB.txt")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_nb, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_NB.tif", overwrite=TRUE)
stopCluster(cl) 
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
