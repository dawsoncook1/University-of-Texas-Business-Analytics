# Import Data
library(readr)
library(kknn) 
library("PerformanceAnalytics")
library(fastDummies)
library(e1071)
LoanData <- read_csv("LoanData.csv")

#Look at the basics of the data
View(LoanData)
summary(LoanData)
attach(LoanData)
#Plot fico vs default

mean(Default)
StdDev(Default)

#Create a histogram showing frequency of fico score vs. frequency of default
hist(fico_range_low[Default==0], #Data of the histogram
     col=5, #Color of the bars
     main="FICO score vs. Default", #Remove title of the graph
     xlab="Fico Score")
hist(fico_range_low[Default==1],
     add=TRUE, #Add the histogram over the previous graph instead of creating a new one
     col=6)
legend("topright", #Position of the legend
       legend=c("Default=0", "Default=1"), #Text of the legend
       fill=c(5,6), #Color of the legend
       bty="n") #Boxtype: n = no box

boxplot(fico_range_low ~ Default, #Formula
        col=c(5,6), #Colors of the boxplots
        horizontal=TRUE,
        main ="FICO vs. Default", #Horizontal boxes
        ylab="Defaulted?",
        xlab="FICO")

#Fitting a linear and logistic regression model
LoanModel = lm(Default~fico_range_low) #Formula (the -1 removes the intercept)
LogLoanModel = glm(Default~fico_range_low, #Formula (the -1 removes the intercept)
                family=binomial) #Family of the glm

par(mfrow=c(2,2))

#Visualizing Linear Regression vs. Logistic Regression
plot(fico_range_low,Default, 
     type="n", 
     main="Linear Regression") 
points(fico_range_low[Default==0],Default[Default==0], 
       col="red", 
       pch=19) 
points(fico_range_low[Default==1],Default[Default==1],col="blue",pch=19)

abline(lm(Default~fico_range_low),
       lwd=2, 
       lty=2) 


ind=order(fico_range_low)
plot(fico_range_low[ind],LogLoanModel$fitted[ind], 
     typ="l", 
     col=1,
     lwd=2, 
     lty=2, 
     ylim=c(0,1), 
     xlab="spread",
     ylab="favwin",
     main="Logistic Regression") #Title of the Graph
points(fico_range_low[Default==0],Default[Default==0],
       col="red",
       pch=19) 
points(fico_range_low[Default==1],Default[Default==1],
       col="blue",
       pch=19) 

#Compare fit of linear vs logistic regression models
summary(LoanModel)
summary(LogLoanModel)

#Now we can explore some different logistic regression models

#Doing some stuff to create a clean data frame with a few new variables
NewLoanData = data.frame(fico_range_low, dti, year,annual_inc, loan_amnt, int_rate )
NewLoanData$loginc = log(NewLoanData$annual_inc)
results <- fastDummies::dummy_cols(NewLoanData, select_columns = "year",)
NewLoanData = data.frame(Default, fico_range_low, dti,loan_amnt, int_rate, results$loginc, results$year_2011, results$year_2012, results$year_2013, results$year_2014)

#Create an expanded logistic regression model that includes more inputs
LogLoanModelplus = glm(Default~fico_range_low + dti + results.loginc + results.year_2011 + results.year_2012 + results.year_2013 + results.year_2014, data = NewLoanData, family = binomial)
summary(LogLoanModelplus)

#Compare to fit of the previous logistic regression model
summary(LogLoanModel)

#Correlation matrix for our independent variables. Everything seems reasonably normal ish
chart.Correlation(NewLoanData, histogram=TRUE, pch=19)

#Note the negative correlation between interest rate and fico score. 

#Now, use stepwise method to choose best model
#Two models initially
null = glm(Default~fico_range_low, data=NewLoanData) #only has an intercept
full = glm(Default~., data=NewLoanData) #Has all the selected variables

#Let us select models by stepwise
regForward = step(null, #The most simple model
                  scope=formula(full),#Let us analyze all models until the full model
                  direction="forward", #Adding variables
                  k=log(length(Default))) #This is BIC
regBack = step(full, #Starting with the full model
               direction="backward", #And deleting variables
               k=log(length(Default))) #This is BIC
regHybrid = step(null, #The most simple model
                  scope=formula(full), #The most complicated model
                  direction="both", #Add or delete variables
                  k=log(length(Default))) #This is BIC

#The stepwise results suggest the following models are the most accurate:
#Model 1 - All variables
#Model 2 - Take out dti
#Model 3 - Take out dti and int_rate


#Cross Validation for the simple and more complex models
library(caret)

set.seed(242)

CVSimple <- train(as.factor(Default)~fico_range_low, NewLoanData,
                  method = 'glm', trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE))

CVModel1 <- train(as.factor(Default)~fico_range_low + dti + int_rate + results.loginc + results.year_2011 + results.year_2012 + results.year_2013 + results.year_2014, NewLoanData,
                  method = 'glm', trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE))

CVModel2 <- train(as.factor(Default)~fico_range_low  + int_rate + results.loginc + results.year_2011 + results.year_2012 + results.year_2013 + results.year_2014, NewLoanData,
                  method = 'glm', trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE))

CVModel3 <- train(as.factor(Default)~fico_range_low  + results.loginc + results.year_2011 + results.year_2012 + results.year_2013 + results.year_2014, NewLoanData,
                  method = 'glm', trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE))
# Accuracy measures for each model after Cross Validation
CVModel1
CVModel2
CVModel3
CVSimple

#From my numbers, CVModel3 is the most accurate at 82.685% accuracy.
#Thus we will leave dti and int_rate out of the model.

Names <- c('Simple', 'Model1', 'Model2', 'Model3')
Accuracy <- c(.771468,.826847,.8266599,.8268468)

barplot(Accuracy, names.arg = Names, ylim = c(.75, .84), col=5)




##############################
####Test best lm on the test set
###############################

train = data.frame(NewLoanData)
test = data.frame(NewLoanData)

set.seed(22)

tr = sample(1:10690, 
            size = 8000, #Approximately 80% into training, 20% into test
            replace = FALSE) 

train = train[tr,] #the rows of train will be the ones sampled
test = test[-tr,] #and test will be everything else (thus, out-of-sample)

######### Using Logistic Regression model to make predictions on test set
######### And printing the confusion matrix
glm.fits2 = lm(Default~fico_range_low  + results.loginc + results.year_2011 + results.year_2012 + results.year_2013 + results.year_2014,data=train)
glm.probs2=predict (glm.fits2,test,type="response")
glm.probs2 [1:10]
glm.pred2=rep('0', 2690)
glm.pred2[glm.probs2 >.5]=" 1"
table(glm.pred2 , test$Default)



###########################################
#Knn Section - Looking at some knn models that might be a good fit and checking for optimal k
###########################################
#Organizing the data in a training df
train = data.frame(fico_range_low,Default)

#Now for the test df
test = data.frame(fico_range_low,Default)
ind = order(test[,1]) #saving the indexes of the first column ordered
test = test[ind,] #Then rewriting the test df with the first column ordered

#Creating the object MSE to be used inside the for loop
MSE = NULL


#Number of neighbors for different models
kk = c(2,10,50,100,150,200,250,300, 1000, 5000)


# WARNING: This line can be memory intensive
for(i in kk){
  #Storing the knn model
  near = kknn(Default ~ fico_range_low, 
              train = train, 
              test = test, 
              k=i, 
              kernel = "rectangular") 
  
  #Calculating the MSE of the current model
  aux = mean((test[,2]-near$fitted)^2)
  
  #Store the MSE of the current model in the MSE object
  MSE = c(MSE,aux)
  
  #Plot a scatterplot
  plot(fico_range_low,Default,main=paste("k=",i),pch=19,cex=0.8,col="darkgray")
  
  lines(test[,1],near$fitted,col=2,lwd=2)
  
  readline("Press [enter] to continue")
}





#Complexity x RMSE graph
plot(log(1/kk),sqrt(MSE), #the values of the graph
     type="b", #Both points and lines
     xlab="Complexity (log(1/k))",
     col="blue", #Color of the line
     ylab="RMSE",
     lwd=2, #line width
     cex.lab=1.2) #Size of lab text

#Text
text(log(1/kk[1]),sqrt(MSE[1])+0.3, #coordinates
     paste("k=",kk[1]), #the actual text
     col=2, #Color of the text
     cex=1.2) #Size of the text
text(log(1/kk[9])+0.4,sqrt(MSE[9]),paste("k=",kk[9]),col=2,cex=1.2)
text(log(1/kk[5])+0.4,sqrt(MSE[5]),paste("k=",kk[5]),col=2,cex=1.2)

# It is obvious from the plot that any k greater than 1000 does not reduce 
# the RMSE by any substantial amount. Thus, choose k = 1000. 

########## Predict knn model on test data

set.seed(22)


train = data.frame(NewLoanData)
test = data.frame(NewLoanData)

tr = sample(1:10690, 
            size = 8000, #Approximately 80% into training, 20% into test
            replace = FALSE) 

train = train[tr,] #the rows of train will be the ones sampled
test = test[-tr,] #and test will be everything else (thus, out-of-sample)

loanknn <- train.kknn(Default ~ fico_range_low, train, kmax = 2)
prediction <- predict(loanknn, test)
knn.pred2=rep('0', 2690)
knn.pred2[prediction >.5]=" 1"

table(test$Default, knn.pred2)

### Knn yields a surprisingly low accuracy of 69.219%


###################################################
## Fit a regression tree to default~fico_range_low.
## The tree is plotted as well as a plot of the corresponding step function
## fit to the data.
###################################################

library(tree)
library(MASS)
library(readr)
#First get a big tree using a small value of mindev (which forces big trees)
temp = tree(as.factor(Default)~fico_range_low, #Formula
            data=NewLoanData, #Data frame
            mindev=.0001) #The within-node deviance must be at least
#this times that of the root node for the node to be split

cat('First create a big tree size: \n')
print(length(unique(temp$where))) #Number of leaf nodes

#Then prune it down to one with 7 leaves
loan.tree=prune.tree(temp, #The tree model
                     best=7) #Only the seven best nodes
cat('Pruned tree size: \n')
print(length(unique(loan.tree$where))) #Number of new leaf nodes


#Plot the tree and the fits.
par(mfrow=c(1,1)) #Plot window: 1 row, 2 columns

#Plot the tree
plot(loan.tree,
     type="uniform") #branches of uniform length
text(loan.tree,pretty=0,col="blue",label=c("yprob"),cex=.8)



################################################################################
## fit boosting for values of
##   (i) depth (ii) number of trees (iii) lamda = shrinkage.
## Fit on train, get loss on validation.
## Write fits on validition from best to file thebpred-2.txt.
################################################################################

ld <- LoanData[,c(3, 7, 17, 26, 29)] # Default,int_rate,dti,loan_amount
#--------------------------------------------------
#train, val, test
set.seed(99)
n=nrow(ld)
n1=floor(n/2)
n2=floor(n/4)
n3=n-n1-n2
ii = sample(1:n,n)
ldtrain=ld[ii[1:n1],]
ldval = ld[ii[n1+1:n2],]
ldtest = ld[ii[n1+n2+1:n3],]
library(gbm) #Boosting package

#Set different sets of parameters for the model
set.seed(1) #Seed to guarantee the same results
idv = c(4,10) #tree depth
ntv = c(100,500) #number of trees
lamv=c(.001,.2) #Learning rates
parmb = expand.grid(idv,ntv,lamv) #Expand the values to get different models
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)

nset = nrow(parmb) #Number of models
olb = rep(0,nset) #Out-of-sample loss
ilb = rep(0,nset) #In-sample loss
bfitv = vector('list',nset) #List of the estimated models

for(i in 1:nset) {
  cat('Model ',i,'out of',nset,'\n')
  tempboost = gbm(as.character(Default)~.,#Formula
                  data=ldtrain, #Data frame
                  distribution='bernoulli',
                  interaction.depth=parmb[i,1], #Maximum depth of each tree
                  n.trees=parmb[i,2], #Number of trees
                  shrinkage=parmb[i,3]) #Learning rate
  ifit = predict(tempboost,n.trees=parmb[i,2],type='response') #In-sample fit
  ofit=predict(tempboost,newdata=ldval,n.trees=parmb[i,2], type='response')
  ifit=round(ifit)
  ofit=round(ofit)
  a=ldval$Default==ofit
  olb=length(a[a==TRUE])/length(a)
  b=ldtrain$Default==ifit
  iib=length(b[b==TRUE])/length(b)
  bfitv[[i]]=tempboost #Saving the model
}
ilb = round(iib,3) #In-sample accuracy
olb = round(olb,3) #IOut-of-sample accuracy

#Print losses
print(cbind(parmb,olb,iib))


#Write val preds
iib=which.min(olb) #Find minimum oos loss
theb = bfitv[[iib]] #Select the model with minimum oos loss
thebpred = predict(theb,newdata=ldval,n.trees=parmb[iib,2]) #Get the prediction for the validation set


################################################################################
## fit random forests on train, get loss on validation
## using values of (i) mtry (ii) number of trees.
## Write out fits on validation from best t file  therfpred.txt.
################################################################################


#BEWARE
rm(list=ls()) #Removes every object from your environment

#Use provided code to get data
LoanData <- read_csv("LoanData.csv")
ld <- LoanData[,c(3, 7, 17, 26, 29)] # Default,int_rate,dti,loan_amount

#--------------------------------------------------
#train, val, test
set.seed(99)
n=nrow(ld)
n1=floor(n/2)
n2=floor(n/4)
n3=n-n1-n2
ii = sample(1:n,n)
ldtrain=ld[ii[1:n1],]
ldval = ld[ii[n1+1:n2],]
ldtest = ld[ii[n1+n2+1:n3],]

library(randomForest)

#Set different sets of parameters for the model
set.seed(1) #Seed to guarantee the same results
p=ncol(ldtrain)-1 #Number of covariates (-1 because one column is the response)
mtryv = c(p,sqrt(p)) #Number of candidate variables for each split
ntreev = c(100,500) #Number of trees
parmrf = expand.grid(mtryv,ntreev) #Expading grids of different models
colnames(parmrf)=c('mtry','ntree')
print(parmrf)

nset = nrow(parmrf) #Number of models
olrf = rep(0,nset) #Out-of-sample loss
ilrf = rep(0,nset) #In-sample loss
rffitv = vector('list',nset) #List of the estimated models

for(i in 1:nset) {
  cat('Model ',i,' out of ',nset,'\n')
  temprf = randomForest(as.factor(Default)~., #Formula
                        data=ldtrain, #Data frame
                        mtry=parmrf[i,1], #Number of candidate variables for each split
                        ntree=parmrf[i,2], #Number of trees
                        maxnodes = 15) #Maximum number of leaves (takes too much time if too big)
  ifit = predict(temprf) #In-sample prediction
  ofit=predict(temprf,newdata=ldval) #Out-of-sample prediction
  
  a=ldval$Default==ofit
  olrf=length(a[a==TRUE])/length(a)
  b=ldtrain$Default==ifit
  ilrf=length(b[b==TRUE])/length(b)
  
  ilrf = round(ilrf,3) #In-sample accuracy
  olrf = round(olrf,3) #Out-of-sample accuracy
}
#Print losses
print(cbind(parmrf,olrf,ilrf))

##The Random Forest model always predicts 0. 

