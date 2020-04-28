
## ROC Plots
library(ROCR)
  
model_1 <- readRDS("Logreg_validation_ROC")
model_2 <- readRDS("TRAIN_GBM_validation_ROC")
preds_list <- list(model_1, model_2)



plot(model_1, col="red")
plot(model_2, add = TRUE, col = "green")
legend(x = "bottomright", 
       legend = c("Model 1", "Model 2"),
       fill = c("red","green"))
