# This script trains a Random Forest model based on the data,
# saves a sample submission

# Download 1_random_forest_r_submission.csv from the output below
# and submit it through https://www.kaggle.com/c/titanic-gettingStarted/submissions/attach
# to enter this getting started competition!
try(setwd("~/../../Downloads"),silent=TRUE)
try(setwd("~/../Downloads"),silent=TRUE)
library(ggplot2)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(ROCR)
library(e1071)

# Read in data
train <- read.csv("train.csv", stringsAsFactors=FALSE)
holdOutTest  <- read.csv("test.csv",  stringsAsFactors=FALSE)
holdOutTestID<-holdOutTest[,1]

# Function to clean the data
extractFeatures <- function(data) {
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  fea$Age[is.na(fea$Age)] <- 0
  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] = "S"
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  return(fea)
}

# Clean data
train<-data.frame(Survived=train[,"Survived"],extractFeatures(train))
holdOutTest<-extractFeatures(holdOutTest)

# Partition training data
set.seed(1000)
inTraining<-createDataPartition(train$Survived,p=0.75,list=FALSE)
training<-train[inTraining,]
testing<-train[-inTraining,]

# Train decision tree model
set.seed(1000)
rpartFit<-train(factor(Survived)~.,
                data=training,
                method="rpart",
                trControl=trainControl(method="cv",
                                       number=10))
rpartPred<-predict(rpartFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],rpartPred)

# Plot ROC curve
pred = prediction(as.numeric(rpartPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train random forest model
set.seed(1000)
rfFit<-train(factor(Survived)~.,
             data=training,
             method="rf",
             trControl=trainControl(method="cv",
                                    number=10))
rfPred<-predict(rfFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],rfPred)

# Plot ROC curve
pred = prediction(as.numeric(rfPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train logistic regression model
set.seed(1000)
glmFit<-train(factor(Survived)~.,
              data=training,
              method="glm",
              trControl=trainControl(method="cv",
                                     number=10))
glmPred<-predict(glmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],glmPred)

# Plot ROC curve
pred = prediction(as.numeric(glmPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train a neural network model
set.seed(1000)
nnFit <- train(factor(Survived)~.,
               data = training, 
               method="nnet",tuneLength=4,maxit=100,trace=F)
nnPred<-predict(nnFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],nnPred)

# Plot ROC curve
pred = prediction(as.numeric(nnPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train SVM Model
set.seed(1000)
svmFit<-svm(factor(Survived)~.,data=training)
svmPred<-predict(svmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],svmPred)

