library(e1071)
library(caret)

setwd("~/Documents/r_wd/machine learning projects/handwriting")

require(data.table)
test <- fread('mnist_test.csv')
train <- fread('mnist_train.csv')

mini_test_indicies <- sample(1:10000, 900)#Sampling 900 from the testing data
mini_train_indicies <- sample(1:60000, 3000)#Sampling 3000 from the training data

mini_test <- test[mini_test_indicies, ]
mini_train <- train[mini_train_indicies, ]

trControl2 <- trainControl(method  = "cv",
                          number  = 1000)

trainX <- mini_train[ , 2:ncol(mini_train)]
trainY <- as.factor(mini_train[['label']])

#Using knn 
knn_fit <- train(trainX,
                 trainY,
                 method = 'knn',
                 trControl = trControl2,
                 metric = 'Accuracy')

x <- predict(knn_fit, newdata = mini_test[ , 2:ncol(mini_test)])
confusionMatrix(x, as.factor(mini_test[['label']]))

#using svm
svm_fit <- svm(x = trainX, 
               y = trainY,
               type = 'C-classification',
               kernel = 'polynomial')

z <- predict(svm_fit, newdata = mini_test[ , 2:ncol(mini_test)])
confusionMatrix(z, as.factor(mini_test[['label']]))

