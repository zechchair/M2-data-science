library(ModelMetrics)
options(digits=20)
library(caTools)
set.seed(37645)
db= read.csv("train_1.csv")
dummy_sep <- rbinom(nrow(db), 1, 0.8)
train = db[dummy_sep == 1, ] 
test = db[dummy_sep == 0, ] 

#train= read.csv("train.csv")
#test=read.csv("test.csv")
########################## split part 
#split = sample.split(db$BPL_B, SplitRatio = 0.50)
#train = subset(db, split==TRUE)
#test = subset(db, split==FALSE)
str(unique(x = train$TRANCHE))
str(unique(x = train$CM))

train=subset(train,select = -c(OBS,RAPPORT_MIN,CM,CMM,OBJECTID,X.1))
test=subset(test,select = -c(OBS,RAPPORT_MIN,CM,CMM,OBJECTID,X.1))

for (valtrain in seq(1,ncol(train),1)) {train[,valtrain] = gsub(",", ".", train[,valtrain])}
for (valtest in seq(1,ncol(test),1)) {test[,valtest] = gsub(",", ".", test[,valtest])}
train$TYPE_=gsub("GEOCHIMIE", "0", train$TYPE_)
train$NIVEAU=gsub("Niveau", "", train$NIVEAU)
train$TYPE_=gsub("MINERALURGIE", "1", train$TYPE_)
train$ZONE_=gsub("Zone", "", train$ZONE_)
train$TRANCHE=gsub("T", "", train$TRANCHE)
train$GISEMENT=gsub("G", "", train$GISEMENT)
train$MINR_PASSANT=gsub("mm", "", train$MINR_PASSANT)
train$MINR_PASSANT=gsub("et", "", train$MINR_PASSANT)
train$MINR_PASSANT = gsub("10   6.3", "8.15", train$MINR_PASSANT)
train$MINR_PASSANT = gsub("10   6.3", "8.15", train$MINR_PASSANT)
train$MINR_PASSANT = gsub("6.3   10", "8.15", train$MINR_PASSANT)

for (coll in seq(1,ncol(train),1)) {
  train[,coll]=as.double(train[,coll])
}

#test cleaning
test$TYPE_=gsub("GEOCHIMIE", "0", test$TYPE_)
test$NIVEAU=gsub("Niveau", "", test$NIVEAU)
test$TYPE_=gsub("MINERALURGIE", "1", test$TYPE_)
test$ZONE_=gsub("Zone", "", test$ZONE_)
test$TRANCHE=gsub("T", "", test$TRANCHE)
test$GISEMENT=gsub("G", "", test$GISEMENT)
test$MINR_PASSANT=gsub("mm", "", test$MINR_PASSANT)
test$MINR_PASSANT=gsub("et", "", test$MINR_PASSANT)
test$MINR_PASSANT = gsub("10   6.3", "8.15", test$MINR_PASSANT)
test$MINR_PASSANT = gsub("10   6.3", "8.15", test$MINR_PASSANT)
test$MINR_PASSANT = gsub("6.3   10", "8.15", test$MINR_PASSANT)


for (coll in seq(1,ncol(test),1)) {
  test[,coll]=as.double(test[,coll])
}
str(train)
str(test)
# Randomly split the data into training and testing sets
set.seed(12)
####################


library(mice)
# Multiple imputation

train = complete(mice(train))

test = complete(mice(test))
cor(train)


