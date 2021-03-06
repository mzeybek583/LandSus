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
Tiff_path <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/savsat/data" 
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

formula <- landslide_non_landslide ~ Altitude + Aspect + Curvature +
Distance_to_drainage + Distance_to_faults + Distance_to_roads + Land_Cover + 
Lithology + Slope_Length + Slope + TWI

library(e1071) 
library(rpart)

#Control NA
sapply(data_df, function(x)sum(is.na(x)))

# Delete NA rows
data_df_NA <- na.omit(data_df)

## Check NA 
sapply(data_df_NA, function(x)sum(is.na(x)))
# data_df_NA <- as.data.frame(lapply(data_df_NA, normalize))
min(data_df_NA$landslide_non_landslide)

#Convert data to levels (factors)
col_names <- names(data_df_NA[,c(-1,-2)])
# do do it for some names in a vector named 'col_names'
#data_df_NA[col_names] <- lapply(data_df_NA[col_names] , numeric)
data_df_NA$landslide_non_landslide <- as.factor(data_df_NA$landslide_non_landslide)
str(data_df_NA)
## Random sampling to from each class at smpl count
model.data <- data.frame()
datalist = list()

for (i in 1:length(levels(data_df_NA$landslide_non_landslide))) {
  out <- data_df_NA[data_df_NA$landslide_non_landslide==i-1,]
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
tiff("RESULT/Model_varimportance_train_LogReg_savsat.tiff", units="cm", width=8, height=8, res=600)
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

saveRDS(gbmImp,"RESULT/Model_varimportance_train_LogReg_savsat")

# Do not change
saveRDS(model_cv, "RESULT/super_model_LogReg_savsat")
# Do not change
library(gmodels)
pred_train <-predict(model_cv, TrainSet[,-10])
pred_valid <-predict(model_cv, ValidSet[,-10])

# Confusion Table Create
confusionMatrix(pred_train, TrainSet[,10], positive="1")
confusionMatrix(pred_valid, ValidSet[,10], positive="1")

# save to file 
# NOT TESTED!!15.06.2020!
cm1<-confusionMatrix(pred_train, TrainSet[,10], positive="1")
cm2<- confusionMatrix(pred_valid, ValidSet[,10], positive="1")
tocsv1 <- data.frame(cbind(t(cm1$overall),t(cm1$byClass)))
tocsv2 <- data.frame(cbind(t(cm2$overall),t(cm2$byClass)))

# You can then use
write.csv(tocsv1,file="RESULT/confusuion_train_savsat.csv")
write.csv(tocsv2,file="RESULT/confusuion_prediction_savsat.csv")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_cv, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_LogReg_savsat.tif", overwrite=TRUE)

#stopCluster(cl)

t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
