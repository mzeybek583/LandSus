# Programmer: Dr. Mustafa ZEYBEK

## ROC Plots
# load the ROC package

library(ROCR)

# Working Directory
setwd(dir = "Code/RESULT/")


# Load ROC data -----------------------------------------------------------


model_1 <- readRDS("TRAIN_Logreg_validation_ROC")
model_2 <- readRDS("ROC_Curve_valid_SVM")


# Plot ROC models ---------------------------------------------------------

png(filename = "../RESULT/ROC_Result.png", width = 400, height = 400)
par(pty="s",cex.axis=1.5, cex.lab=1.3)

plot(model_1, col="red", lwd=1.5, cex.axis=1.5)
plot(model_2, add = TRUE, col = "green", lwd=1.5)
legend(x = "bottomright", 
       legend = c("GLM", "SVM"),
       col = c("red","green"), lty=1, lwd=1.5, cex=1)
abline(a = 0, b = 1, lwd = 1.5, lty = 2)


dev.off()
