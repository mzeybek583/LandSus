# Programmer: Dr. Mustafa ZEYBEK

# Logistic Regression
rm(list = ls()) #Remove everything
# load the raster, sp, and rgdal packages
library(raster)
library(sp)
library(rgdal)
library(caret) 
library(doParallel)

N_CORES <- detectCores()

cl <- makePSOCKcluster(N_CORES-1) # cores
registerDoParallel(cl)

set.seed(917);   

# time 
time <- proc.time()

# DATA --------------------------------------------------------------------
# change
Tiff_path <- "/media/mzeybek/7C6879566879105E/LandslideSusceptibility/Data/savsat/data" 
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

formula <- landslide_non_landslide ~ Altitude + Aspect + Curvature +
  Distance_to_drainage + Distance_to_faults + Distance_to_roads + Land_Cover + 
  Lithology + Slope_Length + Slope + TWI

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
heyelan <- data_df_NA[data_df_NA$landslide_non_landslide==1,]
heyelan_degil <- data_df_NA[data_df_NA$landslide_non_landslide==0,]
data_df_NA_sample_heyelan <- heyelan[sample(1:nrow(heyelan), smpl, replace=FALSE),]
data_df_NA_sample_heyelan_degil <- heyelan_degil[sample(1:nrow(heyelan_degil), smpl, replace=FALSE),]

model.data <- rbind(data_df_NA_sample_heyelan, data_df_NA_sample_heyelan_degil)
#split data into a train and test set 
train <- sample(nrow(model.data), rto*nrow(model.data), replace = FALSE)
TrainSet <- model.data[train,]
ValidSet <- model.data[-train,]
summary(TrainSet)
summary(ValidSet)

## All subsequent models are then run in parallel

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)
# Create Model ------------------------------------------------------------

model_glm <- train(formula, data = TrainSet, 
                   trControl = fitControl, method = "glm",
                   family = "binomial",preProc = c("center", "scale")) 
## Collinearity check

library(car)
library(corrplot)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
p.mat <- cor.mtest(TrainSet[, c(-1,-2,-10)])
corrplot(cor(TrainSet[, c(-1,-2,-10)]), method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         #p.mat = p.mat, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=TRUE ,tl.cex = 0.5, cl.cex = 0.5, number.cex = 0.7
)
png(filename = "corplot_savsat.png")
corrplot(cor(TrainSet[, c(-1,-2,-10)]), method = "number", 
         type = "upper", diag = FALSE, tl.cex = 1.2, cl.cex = 1.2)

dev.off()
model <- lm(formula, TrainSet)
vif(model)
vif(model_glm$finalModel)

summary(model)
proc.time() -time
model_glm

names(model_glm)
model_glm$results
summary(model_glm)

gbmImp <- varImp(model_glm, scale = TRUE)
gbmImp
setwd(file.path(Working_path))

saveRDS(gbmImp,"RESULT/Model_varimportance_train_LogReg_savsat")
#readRDS("Logreg_Train_varimportance")

#png("TRAIN_varImportance_LogReg.png")
tiff("RESULT/Model_varimportance_train_LogReg_savsat.tiff", units="cm", width=8, height=8, res=600)
plot(gbmImp, top = 10)
dev.off()

# Do not change
saveRDS(model_glm, "RESULT/super_model_LogReg_savsat")
# Do not change

library(gmodels)
pred_train <-predict(model_glm, TrainSet[,-10])
pred_valid <-predict(model_glm, ValidSet[,-10])

library(ROCR)
pred <- prediction(predictions = pred_train, labels = TrainSet$landslide_non_landslide)
pred_valid <- prediction(predictions = pred_valid, labels = ValidSet$landslide_non_landslide)

#Confusion Qriteria TRAIN
con_data <- unlist(pred_train)
con_data
con_data[con_data<=0.5] <- 0
con_data[con_data>0.5] <- 1
con_data <- as.factor(con_data)
con_reference <- unlist(TrainSet$landslide_non_landslide)
con_reference <- as.factor(con_reference)
con_mat <- confusionMatrix(data = con_data, reference = con_reference)
con_mat
tocsv <- data.frame(cbind(t(con_mat$overall),t(con_mat$byClass)))
write.csv(tocsv,"RESULT/LogReg_train_confusionMatrix_savsat.csv")

perf <- performance(pred, measure = "tpr", x.measure = "fpr")
perf_valid <- performance(pred_valid, measure = "tpr", x.measure = "fpr")
saveRDS(perf_valid,"RESULT/ROC_Curve_valid_Logreg_savsat")
#aa <- readRDS("Logreg_validation_ROC")
#png("TRAIN_roc_curve_train_LogReg.png")
tiff("RESULT/ROC_Curve_train_LogReg_savsat.tiff", units="cm", width=8, height=8, res=600)
plot(perf, main = "ROC curve for Landslide Detection Train Data (LogReg) Savsat", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
## Export accuracy
dput(perf.auc, "RESULT/Perf_AUC_train_LogReg_savsat.txt")

#png("TRAIN_roc_curve_valid_LogReg.png")
tiff("RESULT/ROC_Curve_valid_LogReg_savsat.tiff", units="cm", width=8, height=8, res=600)

plot(perf_valid, main = "ROC curve for Landslide Detection Validation Data (LogReg) Savsat", col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
dev.off()

perf.auc_valid <- performance(pred_valid, measure = "auc")
str(perf.auc_valid)
unlist(perf.auc_valid@y.values)
## Export accuracy
dput(perf.auc_valid, "RESULT/Perf_AUC_LogReg_savsat.txt")

# Predict raster with produced Super Model --------------------------------
## Apply to raster prediction
r1 <- raster::predict(raster_data, model_glm, progress="text")
plot(r1)

writeRaster(r1,"RESULT/Result_LogReg_savsat.tif", overwrite=TRUE)

stopCluster(cl)

t_end <- proc.time() - time
cat(sprintf("Program Ended in %5.1f second!!!\n", t_end[3]))
  