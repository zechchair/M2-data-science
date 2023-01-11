train_cart=train_lm
test_cart=test_lm


#################### CART

#optimise
library(caret)
library(e1071)

# Number of folds
tr.control = trainControl(method = "cv", number = 100)
# cp values
cp.grid = expand.grid( .cp = (0:1)*0.00001)
# Cross-validation

tr1 = train(BPL_B ~ ., data=train_n0, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
tr2 = train(BPL_B ~., data=train_cart, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
str(train_cart)
best.tree1 = tr1$finalModel
best.tree2 = tr2$finalModel
best.tree.pred1 = predict(best.tree1, newdata=test_n0)
best.tree.pred2 = predict(best.tree2, newdata=test_cart)
rmse(test_n0$BPL_B, best.tree.pred1)
rmse(test_cart$BPL_B, best.tree.pred2)











#library(rpart)
#library(rpart.plot)
#tree1 = rpart(BPL_B ~ F_B+CAO_B+CO2_B+RP+SIO2_B+TYPE_+NIVEAU+GISEMENT+X+Y+PP+ZONE_, data=train_cart,cp=0.0001)
#prp(tree1)
#tree2 = rpart(BPL_B ~ ., data=train_cart)
#tree.pred1 = predict(tree1, newdata=test_cart)
#tree.pred2 = predict(tree2, newdata=test_cart)

r#mse(tree.pred1, test_cart$BPL_B)
r#mse(tree.pred2, test_cart$BPL_B)
