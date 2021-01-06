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

# Calculate the number of cores
num_cores <- detectCores() - 1
registerDoParallel(num_cores)
set.seed(917);   

# time
time <- proc.time() 

# RASTER DATA --------------------------------------------------------------------

Tiff_path <- "D:/data"
Working_path <- "D:/Landsus/LandSus/Code" 
smpl <- 38440 # number of landslide pixels
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
ValidSet <- model.data[-train,]
summary(TrainSet)
summary(ValidSet)
proc.time()- time

## 10-fold cross-validation

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10
                           #classProbs = TRUE,
                           #search = "random"
                           )

# Create Model ------------------------------------------------------------

model_cart <-train(formula, data = TrainSet, 
                  method ='rpart',
                  trControl = fitControl, 
                  #metric = "RMSE",
                  #tuneLength = 10,
                  #preProcess = c("center","scale"), 
                  #allowParallel = TRUE
                  )
getModelInfo(model_cart)

proc.time()- time

plot(model_cart)
model_cart

names(model_cart)
model_cart$results
summary(model_cart)

# Variable Importance
gbmImp <- varImp(model_cart, scale = F)
gbmImp

setwd(file.path(Working_path))

saveRDS(gbmImp,file = "RESULT/Model_varimportance_train_CART")

tiff("RESULT/Model_varimportance_train_CART.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_cart, "RESULT/super_model_CART")
# Do not change

library(gmodels)
pred_train <-predict(model_cart, TrainSet[,-10])
pred_valid <-predict(model_cart, ValidSet[,-10])

# ROC plots ---------------------------------------------------------------

library(ROCR)
pred_training <- prediction(predictions = pred_train, labels = TrainSet$landslide_non_landslide)
pred_validating <- prediction(predictions = pred_valid, labels = ValidSet$landslide_non_landslide)

perf_training <- performance(pred_training, measure = "tpr", x.measure = "fpr")
perf_validating <- performance(pred_validating, measure = "tpr", x.measure = "fpr")

saveRDS(perf_training,"RESULT/ROC_Curve_training_CART")
saveRDS(perf_validating,"RESULT/ROC_Curve_validation_CART")

tiff("RESULT/ROC_Curve_training_CART.tiff", units="cm", width=8, height=8, res=600)
plot(perf_training, main = "Success rate for CART", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred_training, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)

## Export accuracy
dput(perf.auc, "RESULT/Perf_AUC_training_CART.txt")

tiff("RESULT/ROC_Curve_validation_CART.tiff", units="cm", width=8, height=8, res=600)
plot(perf_validating, main = "Prediction rate for CART", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc_valid <- performance(pred_validating, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)

## Export accuracy
dput(perf.auc_valid, "RESULT/Perf_AUC_prediction_CART.txt")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_cart, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_CART.tif", overwrite=TRUE)
t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
