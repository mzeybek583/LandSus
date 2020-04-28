# Programmer: Dr. Mustafa ZEYBEK

## ROC Plots
# load the ROC package

library(ROCR)

# Working Directory
setwd(dir = "/home/mzeybek/LandSus/Code/")

# Load ROC data -----------------------------------------------------------

model_1 <- readRDS("../Code/RESULT/TRAIN_Logreg_validation_ROC")
model_2 <- readRDS("../Code/RESULT/ROC_Curve_valid_SVM")
model_3 <- readRDS("../Code/RESULT/ROC_Curve_valid_RF")

# Plot ROC models ---------------------------------------------------------

png(filename = "../Code/RESULT/ROC_Result.png", width = 400, height = 400)
par(pty="s",cex.axis=1.5, cex.lab=1.3)
plot(model_1, col="red", lwd=1.5, cex.axis=1.5)
text(0.9, 0.5, labels ="AUC: 0.87", col="red")

plot(model_2, add = TRUE, col = "green", lwd=1.5)
text(0.9, 0.4, labels ="AUC: 0.87", col="green")

plot(model_3, add = TRUE, col = "blue", lwd=1.5)
text(0.9, 0.3, labels ="AUC: 0.87", col="blue")
legend(x = "bottomright", 
       legend = c("GLM", "SVM", "RF"),
       col = c("red","green", "blue"), lty=1, lwd=1.5, cex=1)
abline(a = 0, b = 1, lwd = 1.5, lty = 2)

dev.off()
