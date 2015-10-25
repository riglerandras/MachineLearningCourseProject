
## Download and read data

if(!file.exists("pml-testing.csv")){
    download.file(
        url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
        destfile="pml-training.csv")
}

if(!file.exists("pml-testing.csv")){
    download.file(
        url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
        destfile="pml-testing.csv")
}

d <- read.csv("pml-training.csv",row.names=1,na.strings = c("NA",""))
dim(d)

test <- read.csv("pml-testing.csv",row.names=1,na.strings = c("NA",""))
dim(test)


## Partition data

library(caret)
set.seed(123)
inTrain <- createDataPartition(y=d$classe, p=0.7, list=FALSE)
training <- d[inTrain,]
testing <- d[-inTrain,]

rbind(train=dim(training), test=dim(testing))



## Exploratory analysis

table(training$classe)

## check missing values
table(colSums(is.na(training)))

## Remove missing columns from training set

keepCols <- colSums(is.na(training)) == 0
training <- training[,keepCols]
testing <- testing[,keepCols]

## Remove technical columns from training set
training <- training[, -(1:6)]
testing <- testing[, -(1:6)]

## Preprocess with PCA

preProc <- preProcess(training[,-53],method="pca",thresh=.95)
trainPC <- predict(preProc, training)
testPC <- predict(preProc, testing)

modFit <- train(classe ~ ., method = "rf", 
                 data = trainPC, importance = T, 
                 trControl = trainControl(method = "cv", number = 4))

confusionMatrix(predict(modFit, testPC), testPC$classe)


test <- test[,keepCols]
test <- test[, -(1:6)]
testPCsubmit <- predict(preProc, test)

testResults <- predict(modFit, testPCsubmit)

varImp(modFit)
varImpPlot(modFit)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
setwd("./testResults")
pml_write_files(testResults)

## Save model for use in future applications

saveRDS(modFit, "finalModel.Rds")
