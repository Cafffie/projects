library(caret)
library(kernlab)
library(dummy)
library(fastDummies)
library(ROCR)
library(pROC)
library(ROSE)
library(xgboost)
library(e1071)
library(randomForest)


#Loading the dataset
churning_data <- read.csv("./data_Science/churning.csv")
churn2<- churning_data

#Checking the top 5 rows
churning_data[0:5,]

#Exploring the data 
str(churning_data)
dim(churning_data)
summary(churning_data)

#Checking for missing values
colSums(is.na(churning_data))
nrow(churning_data[is.na(churning_data$TotalCharges),])

#Viewing the rows that contains na
churning_data[is.na(churning_data$TotalCharges),]

#Handling missing values (replacing with the mean)
churning_data$TotalCharges[is.na(churning_data$TotalCharges)] <- mean(churning_data$TotalCharges, na.rm=TRUE)
churn2$TotalCharges[is.na(churn2$TotalCharges)] <- mean(churn2$TotalCharges, na.rm=TRUE)

head(churn2)

#Drop unneeded column
churn2 <- churn2[, -1]
str(churn2)

#Select categorical columns
categorical_columns <- churn2 %>%
  select_if(~ !is.numeric(.))
head(categorical_columns)
dim(categorical_columns)

#Select numeric columns
numeric_columns <- churn2 %>%
  select_if(is.numeric)
head(numeric_columns)
dim(numeric_columns)
colSums(is.na(churn2))  

#Viewing unique values in categorical columns
for (col in names(categorical_columns)) {
  print(paste(col, ':', toString(unique(churn2[[col]]))))
}

for (col in names(categorical_columns)){
  print(paste(col, ':', unique(categorical_columns[[col]])))
}

#Replacing values in categorical columns
for (col in names(categorical_columns)) {
  churn2[[col]] <- gsub("No internet service", "No", churn2[[col]])
  churn2[[col]] <- gsub("No phone service", "No", churn2[[col]])
}

#Encoding text in categorical columns to number
for (col in names(categorical_columns)){
  churn2[[col]]  <- gsub("Yes", 1, churn2[[col]])
  churn2[[col]] <- gsub("No", 0, churn2[[col]])
  churn2[[col]] <- gsub("Female", 1, churn2[[col]])
  churn2[[col]] <- gsub("Male", 0, churn2[[col]])
}
churn2$InternetService <- gsub(0, "No", churn2$InternetService)
#churn_with_hyphen <- churn[grepl("-", churn$LAST_12_MONTHS_CREDIT_VALUE), ]

#Creating dummy variables for categorical features
churn2 <- dummy_cols(churn2, select_columns = c("InternetService", "Contract", "PaymentMethod"), remove_first_dummy = TRUE)
head(churn2)

#Dropping repeated columns
churn2 <- churn2[, !names(churn2) %in% c("InternetService", "Contract", "PaymentMethod")]
dim(churn2)
str(churn2)

#Converting all numeric variables to numeric
for (col in names(churn2)){
  if (class(churn2[[col]]) == "character"){
    churn2[[col]] <- as.numeric(churn2[[col]])
  }
}

churn2$SeniorCitizen <- as.numeric(churn2$SeniorCitizen)
churn2$tenure <- as.numeric(churn2$tenure)



#UNIVARIATE ANALYSIS
#Plotting barplots for all categorical columns
for (column in names(categorical_columns)) {
  barplot(table(categorical_columns[[column]]), main= column, col="pink")
}
barplot(table(churn2$SeniorCitizen), main= SeniorCitizen, col="black")

#Distribution plot for continuous variables
#Tenure
ggplot(churn2, aes(x = tenure2)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(tenure2)), color = "black", linetype = "dashed") +
  labs(title = "Tenure2", x = "Tenure2")

#MonthlyCharges
ggplot(churn2, aes(x = MonthlyCharges)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(MonthlyCharges)), color = "black", linetype = "dashed") +
  labs(title = "MonthlyCharges", x = "MonthlyCharges")

#TotalCharges
ggplot(churn2, aes(x = TotalCharges2)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(TotalCharges2)), color = "black", linetype = "dashed") +
  labs(title = "TotalCharges2", x = "TotalCharges2")

#logging the skewed variables to make the distribution normal
churn2$TotalCharges <- log(churn2$TotalCharges)
churn2$tenure <- log(churn2$tenure)


#The data seems imbalanced.
#Using SMOTE to correct it
#Build the model
log1 <- glm(Churn ~ ., data = churn2, family = "binomial")
summary(log1)
log1$fitted.values

#Split the data set into training and test set
churn2$Churn <- as.factor(churn2$Churn)
intrain <- createDataPartition(y=churn2$Churn, p=0.75, list=FALSE)
train <- churn2[intrain,]
test <- churn2[-intrain,]

#Build logistic regression model
log2 <- train(Churn ~., data=train, method="glm", family="binomial")
summary(log2)

#Model prediction
predictions1 <- predict(log2, newdata=test)

#Model accuracy
accuracy1 <- confusionMatrix(predictions1, test$Churn)
print(paste("Accuracy for Logistic Regression model:", accuracy1$overall["Accuracy"]))

Accuracy : 0.8034


#Build a random forest model
library(randomForest)
rf <- train(Churn ~., data=train, method= "rf", 
            trControl=trainControl(method="cv", number=5))
print(rf)

#Predict with the random forest model
predictions2 <- predict(rf, newdata=test)

#Evaluate the model
accuracy2 <- confusionMatrix(predictions2,test$Churn)
print(paste("Accuracy for Random Forest Model:", accuracy2$overall["Accuracy"]))
Accuracy:0.8    


#Build a support vector model
library(e1071)
svm <- train(Churn ~., data=train, method= "svmRadial",
             trControl=trainControl(method="cv", number=5))
print(svm)

#Perform SVM prediction
predictions3 <- predict(svm, newdata=test)
#Evaluate the model
accuracy3 <- confusionMatrix(predictions3, test$Churn)
print(paste("Accuracy for svmRadial:", accuracy3$overall["Accuracy"]))
Accuracy : 0.7932

svm2 <- train(Churn ~., data=train, method="svmLinear",
              trControl= trainControl(method="cv", number=5))
print(svm2)
#Perform SVM prediction
predictions4 <- predict(svm2, newdata=test)

#Evaluate the model
accuracy4 <- confusionMatrix(predictions4, test$Churn)
print(paste("Accuracy for svmLinear:", accuracy4$overall["Accuracy"]))
Accuracy : 0.7847


#Build a K-Nearest Neighbor model
knn <- train(Churn ~., data=train, method="knn",
             trControl=trainControl(method="cv", number = 5),
             tuneGrid= expand.grid(k= c(1, 3, 5))) #specify the "k" values to try

print(knn)
#Model prediction
predictions5 <- predict(knn, test)

#Evaluate the KNN model
accuracy5 <- confusionMatrix(predictions5, test$Churn)
print(paste("Accuracy for KNN model:", accuracy5$overall["Accuracy"]))
Accuracy : 0.7472 


library(xgboost)
#Build the xgboost model
xgboost <- train(Churn ~., data=train, method= "xgbTree",
                 trControl=trainControl(method= "cv", number = 5))
print(xgboost)

#Make predictions
predictions6 <- predict(xgboost, test)

#Evaluate the model
accuracy6 <- confusionMatrix(predictions6, test$Churn)
print(paste("Accuracy for xgboost model:", accuracy6$overall["Accuracy"]))

#Comparing the performance of the six models
print(paste("Accuracy for Logistic Regression model:", accuracy1$overall["Accuracy"]))
print(paste("Accuracy for Random Forest Model:", accuracy2$overall["Accuracy"]))
print(paste("Accuracy for svmRadial:", accuracy3$overall["Accuracy"]))
print(paste("Accuracy for svmLinear:", accuracy4$overall["Accuracy"]))
print(paste("Accuracy for KNN model:", accuracy5$overall["Accuracy"]))
print(paste("Accuracy for xgboost model:", accuracy6$overall["Accuracy"]))





































