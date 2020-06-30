## Set Working Directory

setwd("D:/BABI/Machine Learning/Assesment Project/")
getwd()

library(psych)
library(DataExplorer)
library(corrplot)
library(ppcor)
library(rpart)
library(rpart.plot)
library(ROCR)
library(rattle)
library(car)
library(olsrr)
library(MASS)
library(class)
library(caret)
library(lattice)
library(ggplot2)

#The data is loaded onto mydata object

mydata = read.csv("Cars_edited.csv", header = TRUE)
View(mydata)


#Importing the excel file  the data and descriptive statistics

head(mydata)
dim(mydata)
str(mydata)
names(mydata)
describe(mydata)
summary(mydata)

## Check for missing value (NA)
anyNA(mydata)
plot_missing(mydata)

table(mydata$Transport)

attach(mydata)

## Plotting histogram to understand the overview of the data

#Univariate analysis: Continous variables

plot_histogram(mydata, ggtheme = theme_minimal(base_size = 15))

#Univariate analysis: Numeric variables
colnames(mydata[,sapply(mydata, is.numeric)]) 

ggplot(mydata, aes(x = Age)) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")

boxplot(Age~Transport , main = " Age vs Mode of Transport " , border = "brown")
boxplot(Salary~Transport , main = " Salary vs Mode of Transport " , border = "blue")
boxplot(Distance~Transport , main = " Distance vs Mode of Transport " , border = "green")
boxplot(Work.Exp~Transport , main = " Work Exp vs Mode of Transport " , border = "brown")


ggplot(mydata, aes(x = Salary)) + geom_histogram(bins = 30, fill = "darkgreen", col = "lightgreen")
ggplot(mydata, aes(y = Work.Exp)) + geom_boxplot()

#ggplot(mydata, aes(x = Distance)) + geom_density(col = "blue")
ggplot(mydata, aes(x = Distance)) + geom_histogram(bins = 30, fill = "lightblue", col = "blue")
ggplot(mydata, aes(x = Gender)) + geom_density(col = "darkblue")
ggplot(mydata, aes(y = license)) + geom_boxplot(fill = "lightblue", col = "blue")
ggplot(mydata, aes(x = Engineer)) + geom_histogram(bins = 30, fill = "darkgreen", col = "lightgreen")
ggplot(mydata, aes(x = MBA)) + geom_histogram(bins = 30, fill = "darkgreen", col = "lightgreen")


## Plotting boxplot for all independant variables

boxplot(Age,Work.Exp,Salary,Distance,
        main = "Multiple boxplots for comparision",
        names = c("Age","Work Exp","Salary", "Distance"),
        border = "blue")

table(Gender,Transport)
table(MBA,Transport)
table(Engineer,Transport)
table(licence,Transport)

## Verifying correlation - since the modified excel sheet has multiple factor column
## we cannot perform correleation analysis since correlation in R works mainly on continous 
##variable. So we load the original dataset for corrleation and multicolienarity check .

plot_correlation(mydata)

#Correlation analysis (Corrplot)
library(corrplot)
mydata1= subset(mydata, select = -c(2))
mydata1= subset(mydata1, select = -c(8))
str(mydata1)
cor(mydata1)
datamatrix<-cor(mydata1)
corrplot(datamatrix, method ="number")
# the following are highly correlated: CompRes and DelSpeed; OrdBilling and CompRes; WartyClaim and TechSupport; OrdBilling and DelSpeed; Ecom and SalesFImage.

#Pairwise correlation
# For examining the patterns of multicollinearity, it is required to conduct t-test for correlation coefficient. 
# ppcor package helps to compute the partial correlation coefficients along with the t-statistics and corresponding p values for the independent variables.
library(ppcor)
pcor(mydata1, method = "pearson")
# As expected the correlation between age, salary , work exp  is highly significant; 




#Initial Regression Model using the data as it is
mydata2 = read.csv("Cars_revised.csv", header = TRUE)
View(mydata2)
attach(mydata2)
str(mydata2)
model0 = lm(Transport~., mydata2)
summary(model0)
vif(model0)

# Performing Step regression to remove insignificant variables 
ols_step_both_p(model0, pent = 0.05, prem = 0.3, details = TRUE)

#new model with reduced variables

model1 = lm(Transport~Salary+Age+license+Distance, mydata2)
summary(model1)
vif(model1) # VIF is around 1 for all variables 

## Multicolinearity is removed 
str(mydata2)
cleandata <- subset(mydata2, select = -c(2,3,4,5))
str(cleandata)

## Converting continous to categorical variables

cleandata$Transport = as.factor(cleandata$Transport)
cleandata$license = as.factor(cleandata$license)

anyNA(cleandata)
dim(cleandata)
plot_correlation(cleandata)
names(cleandata)

attach(cleandata)
table(mydata$Transport)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ##SMOTE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

install.packages('DMwR')
library(DMwR)

table(cleandata$Transport)

set.seed(100)

library(caTools) 
split <- sample.split(cleandata$Transport, SplitRatio = .75)

smote.train<-subset(cleandata, split == TRUE)
smote.test<-subset(cleandata, split == FALSE)

TrainData<-subset(cleandata, split == TRUE)
TestData<-subset(cleandata, split == FALSE)

dim(TrainData)
dim(TestData)

table(smote.train$Transport)
table(smote.test$Transport)

smote.train$Transport<-as.factor(smote.train$Transport)
balanced.traindata <- SMOTE(Transport ~., smote.train, perc.over = 350, k = 5, perc.under = 150)
print(balanced.traindata)

#in SMOTE we have to define our equation
#perc.over means that 1 minority class will be added for every value of perc.over
table(balanced.traindata$Transport)

#now we have increased the minority class. We are adding 100 for every minority class sample. - perc.over
#We are subtracting 1 for every 100 - perc.under. We are taking out of the majority class as well.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ##LOGISTIC REGRESSION
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Distribution of minority class in different dataset
prop.table(table(TrainData$Transport))
prop.table(table(TestData$Transport))
prop.table(table(balanced.traindata$Transport))
table(TestData$Transport)
table(smote.test$Transport)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  
str(TrainData)
  
LogitModel1  =  glm(Transport ~ ., 
                     data = TrainData, 
                     family = binomial(link="logit"))

summary(LogitModel1)
vif(LogitModel1)

TestData$log.pred<-predict(LogitModel1, TestData[1:4], type="response")
#in the above line of code we have noted the results of the above two lines in the column called log.pred

Logit1.Accuracy=table(TestData$Transport,TestData$log.pred>0.5)
Logit1.Accuracy
accuracy(Logit1.Accuracy)
#we are comapring the predicted values and given values. Anything above 0.5 will be a yes from the above code

# ROC plot
library(ROCR)
predictROC1 = predict(LogitModel1, newdata = TrainData)
pred1 = prediction(predictROC1, TrainData$Transport)
perf1 = performance(pred1, "tpr", "fpr") 
plot(perf1,colorize =T) 
as.numeric(performance(pred1, "auc")@y.values)
abline(0, 1, lty = 8, col = "blue")

# AUC
train.auc = performance(pred1, "auc")
train.area <- train.auc@y.values[[1]]
train.area

# KS
ks.train <- performance(pred1, "tpr", "fpr")
train.ks <- max(attr(ks.train, "y.values")[[1]] - (attr(ks.train, "x.values")[[1]]))
train.ks

# Gini
train.gini = (2 * train.area) - 1
train.gini

#----------------------------------------------------------------------------------------
LogitModel2  =  glm(Transport ~ ., 
                    data = balanced.traindata, 
                    family = binomial(link="logit"))

summary(LogitModel2)

smote.test$log.pred<-predict(LogitModel2, smote.test[1:4], type="response")
#in the above line of code we have noted the results of the above two lines in the column called log.pred

Logit2.Accuracy=table(smote.test$Transport,smote.test$log.pred>0.5)
#we are comapring the predicted values and given values. Anything above 0.5 will be a yes from the above code

Logit2.Accuracy
accuracy(Logit2.Accuracy)
#we are comapring the predicted values and given values. Anything above 0.5 will be a yes from the above code

# ROC plot
library(ROCR)
predictROC1 = predict(LogitModel2, newdata = TrainData)
pred1 = prediction(predictROC1, TrainData$Transport)
perf1 = performance(pred1, "tpr", "fpr") 
plot(perf1,colorize =T) 
as.numeric(performance(pred1, "auc")@y.values)
abline(0, 1, lty = 8, col = "blue")

# AUC
train.auc = performance(pred1, "auc")
train.area <- train.auc@y.values[[1]]
train.area

# KS
ks.train <- performance(pred1, "tpr", "fpr")
train.ks <- max(attr(ks.train, "y.values")[[1]] - (attr(ks.train, "x.values")[[1]]))
train.ks

# Gini
train.gini = (2 * train.area) - 1
train.gini

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ##KNN
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
library(class)
str(TrainData)

# Normalize variables
scale = preProcess(TrainData, method = "range")

train.norm.data = predict(scale, TrainData)
test.norm.data = predict(scale, TestData)

knn_fit<- knn(train = train.norm.data[,1:4], test = test.norm.data[,1:4], cl= train.norm.data$Transport,k = 3,prob=TRUE)


table(test.norm.data$Transport,knn_fit)
accuracy(table(smote.test$Transport,knn_fit))

#KNN on SMOTE data set 

scale = preProcess(balanced.traindata, method = "range")

train.norm.data = predict(scale, balanced.traindata)
test.norm.data = predict(scale, smote.test)

knn_fit1<- knn(train = train.norm.data[,1:4], test = test.norm.data[,1:4], cl= train.norm.data$Transport,k = 3,prob=TRUE)

table(test.norm.data$Transport,knn_fit1)
accuracy(table(smote.test$Transport,knn_fit1))






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ##Naive Bayes
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TrainData$license = as.numeric(TrainData$license)
TrainData$Transport = as.numeric(TrainData$Transport)

TrainData$license[TrainData$license == 1] <- 0
TrainData$license[TrainData$license == 2] <- 1


TestData$license = as.numeric(TestData$license)
TestData$Transport = as.numeric(TestData$Transport)

TestData$license[TestData$license == 1] <- 0
TestData$license[TestData$license == 2] <- 1


 
library(e1071)
str(TrainData)
str(TestData)
str(train.norm.data)
str(test.norm.data)
nb1 <- naiveBayes(Transport ~., data = train.norm.data)
nbpred<- predict(nb1, newdata = test.norm.data[,1:5])
nbpred
tabnb <- table(TestData$Transport, nbpred)
tabnb
sum(diag(tabnb))/nrow(test)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##Bagging 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

library(xgboost)
library(caret)
library(ipred)
library(rpart)



#we can modify the maxdepth and minsplit if needed
#r doc, https://www.rdocumentation.org/packages/ipred/versions/0.4-0/topics/bagging
str(balanced.traindata)
str(smote.test)
smote.test$Age= as.numeric(smote.test$Age)
smote.test= smote.test[,1:5]

balanced.traindata$license = as.numeric(balanced.traindata$license)
balanced.traindata$Transport = as.numeric(balanced.traindata$Transport)

balanced.traindata$license[balanced.traindata$license == 1] <- 0
balanced.traindata$license[balanced.traindata$license == 2] <- 1

balanced.traindata$Transport[balanced.traindata$Transport == 1] <- 0
balanced.traindata$Transport[balanced.traindata$Transport == 2] <- 1

smote.test$license = as.numeric(smote.test$license)
smote.test$Transport = as.numeric(smote.test$Transport)


smote.test$license[smote.test$license == 1] <- 0
smote.test$license[smote.test$license == 2] <- 1

smote.test$Transport[smote.test$Transport == 1] <- 0
smote.test$Transport[smote.test$Transport == 2] <- 1


bagging1 <- bagging(Transport ~.,
                          data=balanced.traindata,
                          control=rpart.control(maxdepth=5, minsplit=4))


smote.test$pred.class <- predict(bagging1, smote.test)

table(smote.test$Transport,smote.test$pred.class>0.5)#we are comapring our class with our predicted values


#Bagging can help us only so much when we are using a data set that is such imbalanced.

# Boosting

str(features_test)
str(features_train)
str(label_train)
str(balanced.traindata)


install.packages('gbm')
library(gbm)          # basic implementation using AdaBoost
install.packages('xgboost')
library(xgboost)      # a faster implementation of a gbm
install.packages('caret')
library(caret) 


gbm.fit <- gbm(
  formula = Transport ~ .,
  distribution = "bernoulli",#we are using bernoulli because we are doing a logistic and want probabilities
  data = balanced.traindata,
  n.trees = 10000, #these are the number of stumps
  interaction.depth = 1,#number of splits it has to perform on a tree (starting from a single node)
  shrinkage = 0.001,#shrinkage is used for reducing, or shrinking the impact of each additional fitted base-learner(tree)
  cv.folds = 5,#cross validation folds
  n.cores = NULL, # will use all cores by default
  verbose = FALSE#after every tree/stump it is going to show the error and how it is changing
)  
summary(gbm.fit)

smote.test$pred.class <- predict(gbm.fit, smote.test[,1:4], type = "response")

#we have to put type="response" just like in logistic regression else we will have log odds

table(smote.test$Transport,smote.test$pred.class>0.6)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#xgboost()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# XGBoost works with matrices that contain all numeric variables
# we also need to split the training data and label

features_train <- as.matrix(balanced.traindata[,1:4])
label_train <- as.matrix(balanced.traindata[,5])
features_test <- as.matrix(smote.test[,1:4])


xgb.fit <- xgboost(
  data = features_train,
  label = label_train,
  eta = 0.001,#this is like shrinkage in the previous algorithm
  max_depth = 3,#Larger the depth, more complex the model; higher chances of overfitting. There is no standard                      value for max_depth. Larger data sets require deep trees to learn the rules from data.
  min_child_weight = 3,#it blocks the potential feature interactions to prevent overfitting
  nrounds = 10000,#controls the maximum number of iterations. For classification, it is similar to the number of                       trees to grow.
  nfold = 5,
  objective = "binary:logistic",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)


smote.test$xgb.pred.class <- predict(xgb.fit, features_test)

table(smote.test$Transport,smote.test$xgb.pred.class>0.5)
#this model was definitely better
#or simply the total correct of the minority class
sum(smote.test$Transport==1 & smote.test$xgb.pred.class>=0.5)


# Finding the best model (tuning XGB model)
tp_xgb <- vector()
lr <- c(0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1)
md <- c(1, 3, 5, 7, 9, 15)
nr <- c(2, 50, 100, 1000, 10000)

for(i in md) {
  xgb.fit <- xgboost(data = features_train,
                     label = label_train,
                     eta = 0.7,
                     max_depth = 5,
                     min_child_weight = 3,
                     nrounds = 50,
                     nfold = 5,
                     objective = "binary:logistic",
                     verbose = 1,
                     early_stopping_rounds = 10)
  
  smote.test$xgb.pred.class <- predict(xgb.fit, features_test)
  table(smote.test$Transport,smote.test$xgb.pred.class>0.5)
  tp_xgb <- cbind(tp_xgb, sum(smote.test$Transport == 1 & smote.test$xgb.pred.class > 0.5))

}

tp_xgb

##Best XGBoost Model

xgb.fit <- xgboost(data = features_train,
                   label = label_train,
                   eta = 0.7,
                   max_depth = 5,
                   min_child_weight = 3,
                   nrounds = 50,
                   nfold = 5,
                   objective = "binary:logistic",
                   verbose = 1,
                   early_stopping_rounds = 10)

smote.test$xgb.pred.class <- predict(xgb.fit, features_test)
table(smote.test$Transport,smote.test$xgb.pred.class>0.5)
tp_xgb <- cbind(tp_xgb, sum(smote.test$Transport == 1 & smote.test$xgb.pred.class > 0.5))


table(smote.test$Transport, smote.test$xgb.pred.class)


