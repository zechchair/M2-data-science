###################linear reg
library(mice)
# Multiple imputation
test_na$BPL_B=0
test_na$BPL_B=na_if(test_na$BPL_B,0)
train_na$BPL_B=0
train_na$BPL_B=na_if(train_na$BPL_B,0)
train_mice_na=complete(mice(train_na,method='rf'))
test_mice_na=complete(mice(test,method='rf'))


library(Amelia)
test_na_amelia=amelia(test_na,m=5)
train_na_amelia=amelia(train_na,m=5)