library(mlr)
library(h2o)

library(randomForest)
traintask <- makeRegrTask(data = train_n0,target = "BPL_B")
 rf.lrn <- makeLearner("regr.randomForest")
 getParamSet(rf.lrn) 
 params <- makeParamSet(
   makeIntegerParam("mtry",lower = 1,upper = 100),
   makeIntegerParam("nodesize",lower = 1,upper = 50)
 )
 #set validation strategy
 rdesc <- makeResampleDesc("CV",iters=5L)
 ctrl <- makeTuneControlRandom(maxit = 5L)
 tune <- tuneParams(learner = rf.lrn
                    ,task = traintask
                    ,resampling = rdesc
     
                    ,par.set = params
                    ,control = ctrl
                    ,show.info = T)
 
 
 
 
 
  random = randomForest(BPL_B ~ ., data=train_n0,ntree=500,importance=TRUE,mtry=3,nodesize=15 )
 PredictForest = predict(random, newdata = test_n0)
 rmse(PredictForest,test_n0$BPL_B)
 