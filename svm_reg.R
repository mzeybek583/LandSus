  # load the raster, sp, and rgdal packages
  library(raster)
  library(sp)
  library(rgdal)
  
  # set the working directory to the data
  #setwd("pathToDirHere")
  
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
  
  ## class       : RasterStack 
  ## dimensions  : 502, 477, 239454, 3  (nrow, ncol, ncell, nlayers)
  ## resolution  : 1, 1  (x, y)
  ## extent      : 256521, 256998, 4112069, 4112571  (xmin, xmax, ymin, ymax)
  ## coord. ref. : +proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0 
  ## names       : band19, band34, band58 
  ## min values  :     84,    116,    123 
  ## max values  :  13805,  15677,  14343
  
  # plot stack
  plot(raster_data)
  #plotRGB(rgbRaster,r=3,g=2,b=1, stretch = "lin")
  # view histogram of reflectance values for all rasters
  hist(raster_data)
  
  # determine the desired extent
  #rgbCrop <- c(256770.7,256959,4112140,4112284)
  
  # crop to desired extent
  #rgbRaster_crop <- crop(rgbRaster, rgbCrop)
  
  # view cropped stack
  #plot(rgbRaster_crop)
  #plotRGB(rgbRaster_crop,r=3,g=2,b=1, stretch = "lin")
  # create raster brick
  #raster_data_brick <- brick(raster_data)
  
  # check attributes
  #raster_data_brick
  
  # view object size
  #object.size(raster_data_brick)
  
  ## 5759744 bytes
  
  #object.size(raster_data)
  
  ## 41592 bytes
  
  # view raster brick
  #plotRGB(rgbBrick,r=3,g=2,b=1, stretch = "Lin")
  
  # Make a new stack in the order we want the data in 
  #orderRGBstack <- stack(rgbRaster$band58,rgbRaster$band34,rgbRaster$band19)
  
  # write the geotiff
  # change overwrite=TRUE to FALSE if you want to make sure you don't overwrite your files!
  writeRaster(raster_data,"full_data_raster.tif", overwrite=TRUE)
  
  names(raster_data)
  

  data_df <- as.data.frame(raster_data, xy=TRUE)
  rm(raster_data)
  formula <- study_area_heyelan ~ altitude + aspect + corine + curvature + dist_drenaj + dist_fay + jeoloji +  slope + twi+ dist_yol
  #fit = glm(heyelan_training ~ altitude + aspect + corine + curvature + dist_drenaj
   #         + dist_fay + jeoloji +  slope + twi+ dist_yol, family = binomial(), data = data_df)
  
 #class(fit)
 #  fit 
  
  #pred_glm = predict(object = fit, type = "response")
  #head(pred_glm)
  #pred = raster::predict(raster_data, model = fit, type = "response")
  
  #pROC::auc(pROC::roc(raster_data$heyelan_validating, fitted(fit)))
 
  # writeRaster(pred,"pred_glm.tif", overwrite=TRUE)
  
  ## Normalization function
  normalize <- function(x) { return((x - min(x)) / (max(x) - min(x))) }
  
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
  heyelan <- data_df_NA[data_df$study_area_heyelan==1,]
  heyelan_degil <- data_df_NA[data_df_NA$study_area_heyelan==0,]
  data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), 10000,replace=FALSE),]
  data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), 10000,replace=FALSE),]
  
  model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)
  #split data into a train and test set 
  
  train <- sample(nrow(model.data), 0.7*nrow(model.data), replace = FALSE)
  TrainSet <- model.data[train,]
  ValidSet <- model.data[-train,]
  summary(TrainSet)
  summary(ValidSet)
  
library(kernlab)
  #Model building 
  svm.model <- ksvm(formula, data = TrainSet, prob.model=TRUE)
  
  #svm.model <-svm(formula, data = TrainSet, cost =100, gamma =1)
  svm.model
  
  predictions <- predict(svm.model, TrainSet)
  #table(predictions, TrainSet$study_area_heyelan)
  
  #Model evaluation
  library(gmodels)
  svm_pred_train <-predict(svm.model, TrainSet[,-13]) 
  #CrossTable(TrainSet$study_area_heyelan, svm_pred_train, prop.chisq =FALSE, 
  #           prop.c =FALSE, prop.r =FALSE, dnn =c('actual default', 'predicted default')) 
  svm_pred_test <-predict(svm.model, ValidSet[,-13])
  
  #table(TrainSet$study_area_heyelan, svm_pred_train)
  
  library(ROCR)
   pred <- prediction(predictions = svm_pred_train, labels = TrainSet$study_area_heyelan)
   pred_valid <- prediction(predictions = svm_pred_test, labels = ValidSet$study_area_heyelan)
   
   perf <- performance(pred, measure = "tpr", x.measure = "fpr")
   perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
   
   plot(perf, main = "ROC curve for Landslide Detection", col = "blue", lwd = 3)
   abline(a = 0, b = 1, lwd = 2, lty = 2)
   perf.auc <- performance(pred, measure = "auc")
   str(perf.auc)
   unlist(perf.auc@y.values)

   plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data", col = "blue", lwd = 3)
   abline(a = 0, b = 1, lwd = 2, lty = 2)
   perf.auc_valid <- performance(pred_valid, measure = "auc")
   str(perf.auc_valid)
   unlist(perf.auc_valid@y.values)
   
   ## Apply to raster prediction
   raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)
   
   r1 <- raster::predict(raster_data, svm.model, progress="text")
    plot(r1)
   
   writeRaster(r1,"pred_svm2.tif", overwrite=TRUE)
   
   
