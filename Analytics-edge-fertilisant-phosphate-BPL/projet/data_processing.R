library(ModelMetrics)
options(digits=20)
library(caTools)
library(dplyr)  
set.seed(123)
library(mltools)
library(data.table)


db= read.csv("train_1.csv")
#dummy_sep <- rbinom(nrow(db), 1, 0.8)
#train = db[dummy_sep == 1, ] 
#test = db[dummy_sep == 0, ] 
################another method
#ames_split <- initial_split(train, prop = .7)
#ames_train <- training(ames_split)
#ames_test  <- testing(ames_split)



#train= read.csv("train_1.csv")
#test=read.csv("test_1.csv")
########################## split part 
split = sample.split(db$BPL_B, SplitRatio = 0.80)
train = subset(db, split==TRUE)
test = subset(db, split==FALSE)




train=subset(train,select = -c(OBS,RAPPORT_MIN,OBJECTID,CM,CMM,X.1))
test=subset(test,select = -c(OBS,RAPPORT_MIN,OBJECTID,CM,CMM,X.1))
for (valtrain in seq(1,ncol(train),1)) {train[,valtrain] = gsub(",", ".", train[,valtrain])}
for (valtest in seq(1,ncol(test),1)) {test[,valtest] = gsub(",", ".", test[,valtest])}
train$TYPE_=gsub("GEOCHIMIE", "2", train$TYPE_)
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
train$TRANCHE[is.na(train$TRANCHE)]=0
train$ZONE_[is.na(train$ZONE_)]=0
#test cleaning
test$TYPE_=gsub("GEOCHIMIE", "2", test$TYPE_)
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
#test$BPL_B=log(test$BPL_B, base = exp(1))
#train$BPL_B=log(train$BPL_B, base = exp(1))
test$TRANCHE[is.na(test$TRANCHE)]=0
test$ZONE_[is.na(test$ZONE_)]=0
str(train)
str(test)

train_one=train
test_one=test
train_one$TYPE_=as.factor(train_one$TYPE_)
train_one$GISEMENT=as.factor(train_one$GISEMENT)
train_one$TRANCHE=as.factor(train_one$TRANCHE)
train_one$ZONE_=as.factor(train_one$ZONE_)
train_one$MINR_PASSANT=as.factor(train_one$MINR_PASSANT)
train_one$NIVEAU=as.factor(train_one$NIVEAU)
train_one=as.data.frame(one_hot((as.data.table(train_one))))
str(train_one)
test_one$TYPE_=as.factor(test_one$TYPE_)
test_one$GISEMENT=as.factor(test_one$GISEMENT)
test_one$TRANCHE=as.factor(test_one$TRANCHE)
test_one$ZONE_=as.factor(test_one$ZONE_)
test_one$MINR_PASSANT=as.factor(test_one$MINR_PASSANT)
test_one$NIVEAU=as.factor(test_one$NIVEAU)
test_one=as.data.frame(one_hot((as.data.table(test_one))))
test_one[setdiff(names(train_one),names(test_one))]= 0 #create and fill variables which not exist in test by 0
row_sub = apply(train, 1, function(row) all(row !=0 ))
test <- filter(test,BPL_B>0 )
train <- filter(train,BPL_B>0 )


####################


train_na=na_if(train,0)
test_na=na_if(test,0)
table(is.na.data.frame(train))



str(unique(x = train$TRANCHE))
train$TYPE_=gsub("2", "0", train$TYPE_)
test$TYPE_=gsub("2", "0", test$TYPE_)
test[,1]=as.double(test[,1])
train[,1]=as.double(train[,1])
train_non0=subset(train,select = c(TYPE_,NIVEAU,GISEMENT,ZONE_,X,Y,PP,ORDRE,CO2_B,BPL_B))
test_non0=subset(test,select = c(TYPE_,NIVEAU,GISEMENT,ZONE_,X,Y,PP,ORDRE,CO2_B))
table(is.na(train_non0))

#with factors
train_f=train
test_f=test
train_f$TYPE_=as.factor(train_f$TYPE_)
train_f$NIVEAU=as.factor(train_f$NIVEAU)
train_f$GISEMENT=as.factor(train_f$GISEMENT)
train_f$TRANCHE=as.factor(train_f$TRANCHE)
train_f$ZONE_=as.factor(train_f$ZONE_)
train_f$MINR_PASSANT=as.factor(train_f$MINR_PASSANT)
train_f$ORDRE=as.factor(train_f$ORDRE)
for (coll in seq(7,ncol(train_f)-1,1)) {
  train_f[,coll]=as.double(train_f[,coll])
}
test_f$TYPE_=as.factor(test_f$TYPE_)
test_f$NIVEAU=as.factor(test_f$NIVEAU)
test_f$GISEMENT=as.factor(test_f$GISEMENT)
test_f$TRANCHE=as.factor(test_f$TRANCHE)
test_f$ZONE_=as.factor(test_f$ZONE_)
test_f$MINR_PASSANT=as.factor(test_f$MINR_PASSANT)
test_f$ORDRE=as.factor(test_f$ORDRE)
for (coll in seq(7,ncol(test_f)-1,1)) {
  test_f[,coll]=as.double(test_f[,coll])
}
#na values of original table
sapply(train_na, function(x) sum(is.na(x))/length(x))*100
sapply(test, function(x) sum(is.na(x))/length(x))*100
#na values after replacing 0 by na
sapply(train_na, function(x) sum(is.na(x))/length(x))*100
sapply(test_na, function(x) sum(is.na(x))/length(x))*100
#na values of n-zero variables
sapply(train_non0, function(x) sum(is.na(x))/length(x))*100
sapply(test_non0, function(x) sum(is.na(x))/length(x))*100
str(train)
str(train_non0)
str(train_na)
str(train_f)


