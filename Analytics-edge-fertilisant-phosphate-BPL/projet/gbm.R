

train_gbm=train
test_gbm=test

###########gbm

library(gbm)
set.seed(123)
gauss=gbm(BPL_B ~ ., data=train_gbm,distribution="gaussian",n.trees=10000,interaction.depth=4,shrinkage=0.05,verbose=T)

summary(gauss)

predgauss=predict(gauss,newdata=test_gbm,n.trees=10000)
gauss.sse = sum((predgauss - test_gbm$BPL_B)^2)
sqrt(gauss.sse/nrow(test_gbm))


pre=(predgauss+PredictForest)/2
rmse(pre,test_gbm$BPL_B)




par(mfrow=c(1,2))
plot(gauss,i="CO2_B")
plot(gauss,i="ORDRE")
