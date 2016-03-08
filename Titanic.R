####### Titanic ########
getwd()
setwd("~/R/Kaggle/Titanic")

train<-read.csv("train.csv", stringsAsFactors=F, header=T)
test<-read.csv("test.csv", stringsAsFactors=F, header=T)

###### Summary of the data #######
str(train)
summary(train)   ##Age has 177 NAs
barplot(table(train$Survived), names.arg=c("Died", "Survived"))
mosaicplot(train$Sex ~train$Survived, color=T, xlab="Gender", ylab="Survived")
mosaicplot(train$Pclass ~train$Survived, color=T, xlab="Class", ylab="Survived")
mosaicplot(train$Embarked ~train$Survived, color=T, xlab="Class", ylab="Survived")
mosaicplot(train$Embarked ~train$Pclass, color=T, xlab="Embarked", ylab="Class")
boxplot(train$Age~train$Survived, xlab="Survived", ylab="Age")
boxplot(train$Fare~train$Survived, xlab="Survived", ylab="Fare")
boxplot(train$Fare~train$Pclass,xlab="Class", ylab="Fare")
prop.table(table(train$Sex,train$Survived),1)
#Mostly class 3 passengers embark from Queenstown, Cherbourg has the most class 1 passengers

###Segregating Survival by Agegroup and Sex ###
train$AgeGroup <-0
train$AgeGroup[train$Age < 18] <- "Child"
train$AgeGroup[train$Age >= 18 & train$Age <=60] <- "Adult"
train$AgeGroup[train$Age > 60] <- "Elderly"
aggregate(Survived ~ AgeGroup + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
#female elderly most likely to survive, male chidlren greatest chance out of males

###Segregating by Agegroup and Sex and Fare/Class ###
summary(train$Fare)
train$FareType[train$Fare <= 7.91] <- "1st Quartile"
train$FareType[train$Fare >7.91 & train$Fare <= 14.45] <- "2nd Quartile"
train$FareType[train$Fare >14.45 & train$Fare <= 31] <- "3rd Quartile"
train$FareType[train$Fare >31] <- "4th Quartile"
aggregate(Survived ~ Pclass + FareType + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
# no real relationship between faretype and 
aggregate(Survived ~ Pclass + AgeGroup + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
aggregate(Survived ~ FareType + AgeGroup + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

###Looking at titles of passengers
train$title<-sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
train$title<-gsub(" ", "", train$title)
table(train$title)
train$title[train$title %in% c('Capt', 'Don', 'Jonkheer', 'Sir', 'Col', 'Major','Dr', 'Rev')] <- 'Sir'
train$title[train$title %in% c('Mlle')] <- 'Miss'
train$title[train$title %in% c("Mme", "Ms")] <- 'Mrs'
train$title[train$title %in% c("Lady", "theCountess")] <- 'Lady'
prop.table(table(train$title,train$Survived),1)
##Lady and Mrs have higher survival rates than Miss, Master have highest survival rates for males

###Looking at ticket number with letters/ 
TixAlpha<- grep("[^0-9]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixA<- grep("^[A]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixP<- grep("^[P]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixC<- grep("^[C]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixS<- grep("^[S]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixW<- grep("^[W]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixF<- grep("^[F]", train$Ticket, ignore.case=TRUE, value=TRUE)
TixL<- grep("^[L]", train$Ticket, ignore.case=TRUE, value=TRUE)
Tix3<- grep("^[3]", train$Ticket, ignore.case=TRUE, value=TRUE)
Tix1<- grep("^[1]", train$Ticket, ignore.case=TRUE, value=TRUE)
Tix2<- grep("^[2]", train$Ticket, ignore.case=TRUE, value=TRUE)
train$TicketType<- "OtherNumber"
train$TicketType[train$Ticket %in% TixA] <- "A"
train$TicketType[train$Ticket %in% TixP] <- "P"
train$TicketType[train$Ticket %in% TixC] <- "C"
train$TicketType[train$Ticket %in% TixS] <- "S"
train$TicketType[train$Ticket %in% TixW] <- "W"
train$TicketType[train$Ticket %in% TixF] <- "F"
train$TicketType[train$Ticket %in% TixL] <- "Line"
train$TicketType[train$Ticket %in% Tix3] <- "3"
train$TicketType[train$Ticket %in% Tix1] <- "1"
train$TicketType[train$Ticket %in% Tix2] <- "2"
prop.table(table(train$TicketType,train$Survived),1)
### passengers with ticket numbers starting from either 1, or F and P have higher chances of survival. Passengers in A and W have lowest chance

###Looking at family size
train$FamilySize<-train$SibSp + train$Parch
train$FamilyType[train$FamilySize == 0 ] <- "Single"
train$FamilyType[train$FamilySize > 0 & train$FamilySize < 3 ] <- "Small"
train$FamilyType[train$FamilySize >= 3 & train$FamilySize < 6 ] <- "Medium"
train$FamilyType[train$FamilySize >= 6 ] <- "Large"
prop.table(table(train$FamilyType,train$Survived),1)
#smal and medium families have higher chances, large families have the smallest. 


# Lets add missing data
train$Age[is.na(train$Age)]<-median(train$Age,na.rm=TRUE)


# Start ML
train2<-train[,c("Survived","Pclass","Sex", "Age", "FamilyType", "Embarked","SibSp","Parch", "Fare")]
train2$Survived<-as.factor(train$Survived)
train2$Pclass<-as.factor(train$Pclass)
train2$Sex<-as.factor(train$Sex)
train2$FamilyType<-as.factor(train$FamilyType)
train2$Embarked<-as.factor(train2$Embarked)
fit<-randomForest(Survived~.,data=train2)
