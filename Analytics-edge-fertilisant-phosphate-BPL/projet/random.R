train_random=train_one
test_random=test_one
###################random forest
library(mlr)
library(h2o)
library(randomForest)
random = randomForest(BPL_B ~ ., data=train_random,ntree=500,importance=TRUE,mtry=21 )
PredictForest = predict(random, newdata = test_random)

rmse(exp(PredictForest),exp(test_random$BPL_B))


#########BEST 
#The stepFactor specifies at each iteration, mtry is inflated (or deflated) by this value
#The improve specifies the (relative) improvement in OOB error must be by this much for the search to continue
#The trace specifies whether to print the progress of the search
#The plot specifies whether to plot the OOB error as function of mtry

mtry <- tuneRF(train_random[-1],train_random$BPL_B, ntreeTry=50,stepFactor=2,improve=0.3, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

