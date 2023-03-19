install.packages("skimr")
install.packages("tidyverse")
install.packages("mice")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("ggpubr")
install.packages("ROSE")
install.packages("ROCR")
install.packages("randomForest")
install.packages("caret")
install.packages("MLmetrics")

library("skimr")
library("tidyverse")
library("mice")
library("corrplot")
library("ggplot2")
library("ggpubr")
library("ROSE")
library("pROC")
library("randomForest")
library("caret")
library("MLmetrics")


#import the data   Select the location where your data csv is stored
base_data <- read.csv('C:/Users/Administrator/Desktop/report/bank_personal_loan.csv',stringsAsFactors = FALSE)
#Summary of the data
skim(base_data)

#Unified data formats
base_data$Personal.Loan <- as.factor(base_data$Personal.Loan)
base_data$ZIP.Code <- as.factor(base_data$ZIP.Code)
base_data$Education <- as.factor(base_data$Education)
base_data$Online <- as.factor(base_data$Online)
base_data$Securities.Account <- as.factor(base_data$Securities.Account)
base_data$CreditCard <- as.factor(base_data$CreditCard)
base_data$CD.Account <- as.factor(base_data$CD.Account)
sort_name <- c("Personal.Loan","Education","ZIP.Code","Securities.Account","CD.Account","Online","CreditCard","Age","Experience","Income","Family","CCAvg","Mortgage")
base_data <- base_data[,sort_name]
skim(base_data)

#Display missing values   
#There are no missing values in the dataset
md.pattern(base_data)
base_data <- na.omit(base_data)

#numeric 
#show boxplot
boxplot(base_data[,8:13])

#conversion Mortgage to HasMortgage
base_data <- base_data |> 
  mutate(HasMortgage = case_when( Mortgage > 0 ~ 1,TRUE ~ 0)) |> dplyr::select(-Mortgage)
base_data$HasMortgage <- as.factor(base_data$HasMortgage)

#remove Outliers
base_data <- base_data[-which(base_data$Income %in% boxplot.stats(base_data$Income)$out),]

#Correlation analysis 
cor1 <- cor(base_data[,8:12])
corrplot(cor1, method="number") 


#factor
p1 <- ggplot(base_data, aes(x=Personal.Loan))+geom_bar()
p2 <- ggplot(base_data, aes(x=Education))+geom_bar()
p4 <- ggplot(base_data, aes(x=Securities.Account))+geom_bar()
p5 <- ggplot(base_data, aes(x=CD.Account))+geom_bar()
p6 <- ggplot(base_data, aes(x=Online))+geom_bar()
p7 <- ggplot(base_data, aes(x=CreditCard))+geom_bar()
ggarrange(p1,p2,p4,p5,p6,p7)

#Feature selection 
fcdata <- base_data[,c(8:12)]
fc <- princomp(base_data[,c(8:12)], cor = TRUE, scores = TRUE)
plot(fc, type="lines")
summary(fc)

#model data
select_f <- c("Personal.Loan","Education","Securities.Account","CD.Account","Online","CreditCard","HasMortgage","Age","Income","Family","CCAvg")
base_modeldata <- base_data[,select_f]
#write.csv(base_modeldata,"C:/Users/Administrator/Desktop/R/base_modeldata.csv")

#Train-test split
set.seed(150)
splitindex <- createDataPartition(base_modeldata$Personal.Loan,times = 1,p=0.7,list = FALSE)
train <- base_modeldata[splitindex,]
test <- base_modeldata[-splitindex,]
table(train$Personal.Loan)
table(test$Personal.Loan)


#split 10-cv data Ensure data unity
k_data <- createFolds(y=base_modeldata$Personal.Loan, k=10,list = TRUE, returnTrain = TRUE)
print(base_modeldata[k_data[[1]],])


#Simple logistic regression model
model_1 <- glm(Personal.Loan~.,data=train, family="binomial")
summary(model_1)
probs_1 <- predict(model_1, test, type="response")
model1.pred <- ifelse(probs_1 > 0.5, 1, 0)
table(truth=test$Personal.Loan,prediction=model1.pred)
f1_1 <- F1_Score(test$Personal.Loan, model1.pred, positive = "1")
print(f1_1)
#0.7391304

#ROC
modelroc <- roc(test$Personal.Loan,probs_1)
plot(modelroc,print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE)

#Confusion Matrix
ctable1 <- as.table(caret::confusionMatrix(factor(test$Personal.Loan),factor(model1.pred)))
fourfoldplot(ctable1,color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Logistic Regression Confusion Matrix")


#10 cross-validation
glm.auc <- as.numeric()
glm.acc <- glm.f1 <- c()
for(i in 1:10){
  fold_test <- base_modeldata[k_data[[i]],]
  fold_train <- base_modeldata[-k_data[[i]],]
  
  glm.model <- glm(Personal.Loan~.,fold_train,family="binomial")
  glm.pred = predict(glm.model, fold_test, type="response") 
  glm.pred <- ifelse(glm.pred > 0.5, 1, 0)
  
  glm.f1[i] <- F1_Score(fold_test$Personal.Loan, glm.pred, positive = "1")
  glm.acc[i] <- sum(glm.pred == fold_test$Personal.Loan)/nrow(fold_test)
  modelroc3 = roc(fold_test$Personal.Loan, glm.pred)
  glm.auc <- append(glm.auc, auc(modelroc3))
}

glm.acc.train <- round(mean(glm.acc), 5) * 100
glm.f1.train <- round(mean(glm.f1), 5) * 100
glm.auc.train <- round(mean(glm.auc), 5) * 100

print(glm.acc.train)
print(glm.f1.train)
print(glm.auc.train)


#random forest
#10 cross-validation
rf.grid <- expand.grid(nt = seq(100,500,by=100), mtry=c(1,3,5,7,10))
rf.grid$acc <- rf.grid$f1 <- NA

rf.f1 <- rf.acc <- c()

for (k in 1:nrow(rf.grid)) {
  for (i in 1:10) {
    fold_test1 <- base_modeldata[k_data[[i]],]
    fold_train1 <- base_modeldata[-k_data[[i]],]
    
    rf.model <- randomForest(Personal.Loan~., data = fold_train1, 
                             ntree = rf.grid$nt[k],
                             mtry = rf.grid$mtry[k])
    rf.pred <- predict(rf.model, fold_test1)
    
    rf.f1[i] <- F1_Score(fold_test1$Personal.Loan, rf.pred, positive = "1")
    rf.acc[i] <- sum(rf.pred == fold_test1$Personal.Loan)/nrow(fold_test1)
  }
  rf.grid$f1[k] <- mean(rf.f1)
  rf.grid$acc[k] <- mean(rf.acc)
}

rf.grid[which.max(rf.grid$f1),]
#300    7 0.8831426 0.9805827

#Model comparison
model_2 <- randomForest(Personal.Loan~.,data = train, ntree = 300, mtry =7)
probs_2 <- predict(model_2, test)

f1_2 <- F1_Score(test$Personal.Loan, probs_2, positive = "1")
print(f1_2)
#0.879668

#ROC
roc_rf <- roc(test$Personal.Loan, as.numeric(probs_2))
plot(roc_rf,print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE)

table(truth=test$Personal.Loan,prediction=probs_2)
#Confusion Matrix
ctable2 <- as.table(confusionMatrix(test$Personal.Loan,
                         factor(probs_2)))
fourfoldplot(ctable2,color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Random Forest Confusion Matrix")


#Feature importance 
model_2$importance
importance <- as.data.frame.table(model_2$importance, keep.rownames = TRUE)
colnames(importance) <- c("Feature", "Importance","value")
ggplot(importance, aes(x=reorder(Feature,Importance),y=value))+geom_col()+coord_flip()+xlab("")

rf2.grid <- expand.grid(nt = seq(100,500,by=100), mtry=c(1,3,5,7,10))
rf2.grid$acc <- rf2.grid$f1 <- NA

rf2.f1 <- rf2.acc <- c()

for (k in 1:nrow(rf2.grid)) {
  for (i in 1:10) {
    fold_test2 <- base_modeldata[k_data[[i]],]
    fold_train2 <- base_modeldata[-k_data[[i]],]
    
    rf.mode2 <- randomForest(Personal.Loan~Education+Income+Family+CCAvg, data = fold_train2, 
                             ntree = rf2.grid$nt[k],
                             mtry = rf2.grid$mtry[k])
    rf.pred2 <- predict(rf.mode2, fold_test2)
    
    rf2.f1[i] <- F1_Score(fold_test2$Personal.Loan, rf.pred2, positive = "1")
    rf2.acc[i] <- sum(rf.pred2 == fold_test2$Personal.Loan)/nrow(fold_test2)
  }
  rf2.grid$f1[k] <- mean(rf2.f1)
  rf2.grid$acc[k] <- mean(rf2.acc)
}

rf2.grid[which.max(rf2.grid$f1),]
#300    3 0.8807026 0.9800389

model_3 <- randomForest(Personal.Loan~Education+Income+Family+CCAvg,data = train, ntree = 300, mtry =3)
probs_3 <- predict(model_3, test)

f1_3 <- F1_Score(test$Personal.Loan, probs_3, positive = "1")
print(f1_3)
#0.8825911

roc_rf2 <- roc(test$Personal.Loan, as.numeric(probs_3))
plot(roc_rf2,print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE)

ctable3 <- as.table(confusionMatrix(test$Personal.Loan,
                                    factor(probs_3)))
fourfoldplot(ctable3,color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Random Forest Confusion Matrix")


