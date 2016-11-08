
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
extractFeatures <- function(data,inclSurvived=FALSE) {
  
  # Create Titles
  data$Title<-NA
  data$Title[grep("Mr\\.",data$Name)]<-"MR"
  data$Title[grep("Mrs\\.|Mme\\.|Lady|Countess\\.",data$Name)]<-"MRS"
  data$Title[grep("Miss|Mlle\\.|Ms\\.",data$Name)]<-"MS"
  data$Title[grep("Master",data$Name)]<-"MASTER"
  data$Title[grep("Rev\\.",data$Name)]<-"REV"
  data$Title[grep("Dr\\.",data$Name)]<-"DR"
  data$Title[grep("Major\\.|Col\\.|Capt\\.",data$Name)]<-"OFFICER"
  data$Title[grep("Sir\\.|Don\\.|Jonkheer\\.",data$Name)]<-"SIR"
  
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked",
                "Title")
  if (inclSurvived==TRUE) {
    fea <- data[,c("Survived",features)]
    fea$Survived[fea$Survived==0]<-"Died"
    fea$Survived[fea$Survived==1]<-"Survived"
  } else {
    fea <- data[,features]  
  }

  # fea$Age[is.na(fea$Age)] <- 0
  # fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] <- NA
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  fea<-fea[complete.cases(fea),]
  
  # Feature creation
  
  return(fea)
}

# Clean data
train<-extractFeatures(train,inclSurvived=TRUE)
holdOutTest<-extractFeatures(holdOutTest)

# Partition training data
set.seed(1000)
inTraining<-createDataPartition(train$Survived,p=0.75,list=FALSE)
training<-train[inTraining,]
testing<-train[-inTraining,]

# Train Naives Bayes model
set.seed(1000)
nbFit<-train(factor(Survived)~.,
             data=training,
             method="nb",
             trControl=trainControl(method="repeatedcv",
                                    number=10,
                                    repeats=10))
nbPred<-predict(nbFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],nbPred)

# Train decision tree model
set.seed(1000)
rpartFit<-train(factor(Survived)~.,
                data=training,
                method="rpart",
                trControl=trainControl(method="repeatedcv",
                                       number=10,
                                       repeats=10))
rpartPred<-predict(rpartFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],rpartPred)

# Plot ROC curve
pred = prediction(as.numeric(rpartPred), as.numeric(testing[,c("Survived")]))
roc = performance(pred, measure="tpr", x.measure="fpr")
plot(roc, col="orange", lwd=2) 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=2)
auc = performance(pred, 'auc')
slot(auc, 'y.values')

# Train a boosted tree model
set.seed(1000)
gbmGrid<-expand.grid(interaction.depth=c(1,5,9),n.trees=(1:30)*50,shrinkage=0.1,n.minobsinnode=20)
gbmFit<-train(factor(Survived)~.,
              data=training,
              method="gbm",
              trControl=trainControl(method="repeatedcv",
                                     number=10,
                                     repeats=10,
                                     summaryFunction=twoClassSummary,
                                     classProbs=TRUE,
                                     allowParallel=TRUE),
              verbose = FALSE,metric="ROC",tuneGrid=gbmGrid)
gbmPred<-predict(gbmFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],gbmPred)

# Train random forest model
set.seed(1000)
rfFit<-train(factor(Survived)~.,
             data=training,
             method="rf",
             trControl=trainControl(method="repeatedcv",
                                    number=10,
                                    repeats=10))
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
              trControl=trainControl(method="repeatedcv",
                                     number=10,repeats=10))
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


# Train SVM with Radial Kernel Model
set.seed(1000)
svmRadialFit<-train(factor(Survived)~.,
                    data=training,
                    method="svmRadial",
                    trControl=trainControl(method="repeatedcv",
                                           number=10,
                                           repeats=10,
                                           summaryFunction=twoClassSummary,
                                           classProbs=TRUE,
                                           allowParallel=TRUE),
                    preProc = c("center","scale"),
                    metric="ROC")
svmRadialPred<-predict(svmRadialFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],svmRadialPred)

# Train Adaboost Model
set.seed(1000)
adaFit<-train(factor(Survived)~.,
                    data=training,
                    method="adaboost",
                    trControl=trainControl(method="repeatedcv",
                                           number=10,
                                           repeats=10,
                                           summaryFunction=twoClassSummary,
                                           classProbs=TRUE,
                                           allowParallel=TRUE),
                    metric="ROC")
adaPred<-predict(adaFit,newdata=testing)
confusionMatrix(testing[,c("Survived")],adaPred)