# Programmer: Dr. Mustafa ZEYBEK

## ROC Plots
# load the ROC package

library(ROCR)
library(stringr)

numextract <- function(string){ 
  str_extract(string, "\\-*\\d+\\.*\\d*")
} 
# Working Directory
setwd(dir = "/home/mzeybek/LandSus/Code/")

# Load ROC data -----------------------------------------------------------

model_1 <- readRDS("../Code/RESULT/TRAIN_Logreg_validation_ROC")




model_2 <- readRDS("../Code/RESULT/ROC_Curve_valid_SVM")
model_3 <- readRDS("../Code/RESULT/ROC_Curve_valid_RF")

auc1 <- read.csv("RESULT/Perf_AUC_LogReg.txt")
auc1 <- auc1$y.name...Area.under.the.ROC.curve
auc1 <- as.numeric(numextract(auc1)[1])
auc1 <- format(auc1, digits = 3)

auc2 <- read.csv("RESULT/Perf_AUC_SVM.txt")
auc2 <- auc2$y.name...Area.under.the.ROC.curve
auc2 <- as.numeric(numextract(auc2)[1])
auc2 <- format(auc2, digits = 3)

auc3 <- read.csv("RESULT/Perf_AUC_RF.txt")
auc3 <- auc3$y.name...Area.under.the.ROC.curve
auc3 <- as.numeric(numextract(auc3)[1])
auc3 <- format(auc3, digits = 3)

# Plot ROC models ---------------------------------------------------------

png(filename = "../Code/RESULT/ROC_Result.png", width = 400, height = 400)
par(pty="s",cex.axis=1.5, cex.lab=1.3)
plot(model_1, col="red", lwd=1.5, cex.axis=1.5)
text(0.9, 0.5, labels = paste("AUC:", auc1), col="red")

plot(model_2, add = TRUE, col = "green", lwd=1.5)
text(0.9, 0.4, labels = paste("AUC:", auc2), col="green")

plot(model_3, add = TRUE, col = "blue", lwd=1.5)
text(0.9, 0.3, labels = paste("AUC:", auc3), col="blue")
legend(x = "bottomright", 
       legend = c("GLM", "SVM", "RF"),
       col = c("red","green", "blue"), lty=1, lwd=1.5, cex=1)
abline(a = 0, b = 1, lwd = 1.5, lty = 2)

dev.off()
