

Reg_FB=lm(BPL_B ~ F_B, data=train_na)
summary(Reg_FB)


train_cart=train_na
test_cart=test_na


#################### CART

#optimise
library(caret)
library(e1071)

# Number of folds
tr.control = trainControl(method = "cv", number = 10)
# cp values
cp.grid = expand.grid( .cp = (0:1)*0.00001)
# Cross-validation

tr1 = train(BPL_B ~ ., data=train_cart, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
str(train_cart)
best.tree1 = tr1$finalModel

best.tree.pred1 = predict(best.tree1, newdata=test_cart)

rmse(test_cart$BPL_B, best.tree.pred1)












Reg.pred=c()
rms=c()
for (i in seq(1,nrow(test_na),1)){
    if (!is.na(test_na$F_B[i])){
Reg.pred[i] = predict(Reg_FB, newdata=test_na[i,])

    }else if (is.na(test_na$F_B[i])) {Reg.pred[i]=(predgauss[i]+PredictForest[i])/2
   }
}
rmse(Reg.pred, test_gbm$BPL_B)

