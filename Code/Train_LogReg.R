        
        # Logistic Regression
#REmove everything
rm(list = ls())
        # load the raster, sp, and rgdal packages
        library(raster)
        library(sp)
        library(rgdal)
        library(caret) 
       # library(randomForest) 
       # library(doParallel)
        
        # registerDoParallel(cores = 3)
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

        # plot stack
       # plot(raster_data)
        #plotRGB(rgbRaster,r=3,g=2,b=1, stretch = "lin")
        # view histogram of reflectance values for all rasters
        #hist(raster_data)
        
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
        #writeRaster(raster_data,"full_data_raster.tif", overwrite=TRUE)
        
        names(raster_data)
        data_df <- as.data.frame(raster_data, xy=TRUE)
        rm(raster_data)
        formula <- study_area_heyelan ~ altitude + aspect + corine + curvature + dist_drenaj + dist_fay + jeoloji +  slope + twi + dist_yol
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
      # data_df_NA <- as.data.frame(lapply(data_df_NA, normalize))
        
        
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

## All subsequent models are then run in parallel
        #train_control <- trainControl(method="LOOCV")
        ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
#model_glm <- glm(formula, data = TrainSet, family=binomial, trControl=train_control)
        #model_glm <- train(formula, data = TrainSet, method = "bayesglm", family=binomial, trControl=train_control)
       #model_glm <- train(formula, data = TrainSet, 
       #                    trControl = trainControl(method = "cv", number = 5),
       #                    method = "glm",
       #                   family = "binomial") 
model_glm <- train(formula, data = TrainSet, 
                   trControl = ctrl, method = "glm",
                   family = "binomial",preProc = c("center", "scale")) 
## When you are done:
proc.time() -time
model_glm
#trainControl(method = "cv", number = 5)[1:3]
names(model_glm)
model_glm$results
summary(model_glm)
# calc_acc = function(actual, predicted) {
#         mean(actual == predicted)
# }
# calc_acc(actual = TrainSet$study_area_heyelan,
#          predicted = predict(model_glm, newdata = TrainSet))

gbmImp <- varImp(model_glm, scale = TRUE)
gbmImp
saveRDS(gbmImp,"Logreg_Train_varimportance")
#readRDS("Logreg_Train_varimportance")

        #png("TRAIN_varImportance_LogReg.png")
tiff("TRAIN_varImportance_LogReg.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_glm, "super_model_LogReg")
# Do not change

library(gmodels)
pred_train <-predict(model_glm, TrainSet[,-13])
pred_valid <-predict(model_glm, ValidSet[,-13])



library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$study_area_heyelan)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$study_area_heyelan)

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"TRAIN_Logreg_validation_ROC")
#aa <- readRDS("Logreg_validation_ROC")
#png("TRAIN_roc_curve_train_LogReg.png")
tiff("TRAIN_roc_curve_train_LogReg.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (LogReg)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "TRAIN_LogReg.txt")

#png("TRAIN_roc_curve_valid_LogReg.png")
tiff("TRAIN_roc_curve_valid_LogReg.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (LogReg)", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()


perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "TRAIN_valid_LogReg.txt")


## Apply to raster prediction
# raster_data <- stack(altitude, aspect, corine, curvature, drenaj, fay, jeoloji, slope, twi, yol, cls)
# 
# r1 <- raster::predict(raster_data, model_glm, progress="text")
# plot(r1)
# 
# writeRaster(r1,"TRAIN_LogReg.tif", overwrite=TRUE)
proc.time() - time