library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization


set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = BPL_B ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 10000,
  interaction.depth = 4,
  shrinkage = 0.02,
  cv.folds = 5,
  n.cores =NULL , # will use all cores by default
  verbose = FALSE
)  


# print results
print(gbm.fit)
# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))
## [1] 29133.33
predgauss=predict(gbm.fit,newdata=test,n.trees=10000)

RMSE(predgauss,test$BPL_B)
# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")
# create hyperparameter grid
hyper_grid <- expand.grid(
  trees = seq(5000,10000,1000),
  shrinkage = seq(0.01,0.2,0.05),
  interaction.depth = seq(1,20,5),
n.minobsinnode = seq(1,50,9),
 bag.fraction = c(0.6,0.5), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)
## [1] 81
# randomize data
random_index <- sample(1:nrow(train), nrow(train))
random_train <- train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = BPL_B ~ .,
    distribution = "gaussian",
    data = random_train,
    n.trees = hyper_grid$trees[i],
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
#    n.minobsinnode = hyper_grid$n.minobsinnode[i],
#    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
cv.folds = 5,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}
#head(hyper_grid,n=100)
head(hyper_grid %>% dplyr::arrange(min_RMSE),n=10)

