# Programmer: Dr. Mustafa ZEYBEK

## Neural Network nnet


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

## All subsequent models are then run in parallel

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

#Train and Tune the SVM

model_nnet <- train(formula, data = TrainSet,
                  method = "nnet",
                  preProc = c("center","scale"),
                  #tuneGrid = grid,
                  trControl= fitControl,
                  nodesize=100,
                  trace=FALSE) 
proc.time()- time

model_nnet
names(model_nnet)
model_nnet$results
summary(model_nnet)

gbmImp <- varImp(model_nnet, scale = TRUE)
gbmImp

setwd(file.path(Working_path))

saveRDS(gbmImp,"RESULT/Model_varimportance_train_NNET_class")
#readRDS("SVM_Train_varimportance")

#png("TRAIN_varImportance_SVM.png")
tiff("RESULT/Model_varimportance_train_NNET_v2.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_nnet, "RESULT/super_model_NNET_v2")
# Do not change

library(gmodels)
pred_train <-predict(model_nnet, TrainSet[,-12])
pred_valid <-predict(model_nnet, ValidSet[,-12])

# Confusion Table Create
# https://www.wikiwand.com/en/Confusion_matrix
confusionMatrix(pred_train, TrainSet[,12], positive="1")
confusionMatrix(pred_valid, ValidSet[,12], positive="1")

# save to file 
# NOT TESTED!!15.06.2020!
cm1<-confusionMatrix(pred_train, TrainSet[,12], positive="1")
cm2<- confusionMatrix(pred_valid, ValidSet[,12], positive="1")
tocsv1 <- data.frame(cbind(t(cm1$overall),t(cm1$byClass)))
tocsv2 <- data.frame(cbind(t(cm2$overall),t(cm2$byClass)))

# You can then use
write.csv(tocsv1,file="RESULT/confusuion_train.csv")
write.csv(tocsv2,file="RESULT/confusuion_prediction.csv")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_nnet, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_NNET_v2.tif", overwrite=TRUE)

#stopCluster(cl)

t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
