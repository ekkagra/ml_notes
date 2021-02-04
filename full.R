###----- REGRESSION --------###
# ------ Linear Regression --- #
# ---------------------------- #
dataset = read.csv('Salary_Data.csv')

library(caTools)

set.seed(10283)
split = sample.split(dataset$Salary,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(Salary ~ YearsExperience, data = dataset)

y_pred = predict(regressor,newdata = test_set)


library(ggplot2)
ggplot() + 
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),colour='red')+
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),colour='blue')

ggplot() +
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),colour='red')+
  geom_line(aes(x=test_set$YearsExperience,y=predict(regressor,newdata = test_set)),colour='blue')

# ---------------------------------------- #
# --------- Multinomial Regression ------- #
# ---------------------------------------- #
library(caTools)
dataset = read.csv('50_Startups.csv')
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset,split=TRUE)
test_set = subset(dataset,split=FALSE)

regressor = lm(formula = Profit~.,data=training_set)
summary(regressor)

y_pred = predict(regressor,newdata= test_set)

# --------------------------------------- #
# --------- Polynomial Regression ------- #
# --------------------------------------- #
dataset = read.csv('Position_Salaries.csv')

dataset = dataset[2:3]

# lin_reg = lm(formula = Salary ~ .,
#              data = dataset)
# library(ggplot2)
# ggplot()+
#   geom_point(aes(x=dataset$Level,y=dataset$Salary),colour='red') +
#   geom_line(aes(x=dataset$Level,y=predict(lin_reg,newdata=dataset)),colour='blue')
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~.,data = dataset)

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),colour='red') +
  geom_line(aes(x=dataset$Level,y=predict(poly_reg,newdata=dataset)),colour='blue')

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue')

predict(lin_reg,data.frame(Level=6.5))
a <- 7.718
predict(poly_reg,data.frame(Level=a,
                            Level2=a^2,
                            Level3=a^3,
                            Level4=a^4))
summary(poly_reg)
summary(regressor)
# ---------------------------------- #
# ---------- SVR Regression -------- #
# ---------------------------------- #
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')
y_pred = predict(regressor, data.frame(Level = 6.5))

library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),colour='red')+
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata = dataset)),
            colour='blue')

# ----------------------------------------- #
# --------- Decision Tree Regression ------ #
# ----------------------------------------- #
dataset = read.csv('Position_Salaries.csv')
dataset <- dataset[2:3]

library(rpart)
regressor = rpart(formula = Salary~.,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

y_pred = predict(regressor, data.frame(Level = 6.5))

library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# -------------------------------------- #
# ------ RandomForest Regressor -------- #
# -------------------------------------- #
library(randomForest)
regressor_rf = randomForest(x=dataset[-2],
                            y=dataset$Salary,
                            ntree = 500)
y_pred = predict(regressor_rf,newdata = data.frame(Level=6.5))

library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),colour='red')+
  geom_line(aes(x=x_grid,y=predict(regressor_rf,newdata = data.frame(Level=x_grid)))
            ,colour = 'blue')


# ------- Regression Evaluation
# Mean Absolute Error 
# Mean Squared Error
# R2 Score
summary(regressor)


# ------ Iris Data
data("iris")
set.seed(1)
training_idx = sample(1:nrow(iris),nrow(iris)*0.8,replace = FALSE)
holdout_idx = sample(1:nrow(iris),training_idx)
training = iris[training_idx,]
holdout = iris[holdout_idx,]

m = lm(training$Sepal.Length~.,training)
summary(m)
training_res = training$Sepal.Length - predict(m,training)
holdout_res = holdout$Sepal.Length - predict(m,holdout)

# -------------------------------------- #
###--------- CLASSIFICATION ---------- ###
# -------------------------------------- #

#------- Logistic Regression
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset, split == FALSE)

training_set[-3] = scale(training_set[-3])
test_set[-3]=scale(test_set[-3])

classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

cm_log = table(test_set[,3],y_pred)

#------- KNN Classifier
library(class)
y_pred_knn = knn(train = training_set[,-3],
                 test = test_set[,-3],
                 cl = training_set[,3],
                 k=5,
                 prob = TRUE)

cm_knn = table(test_set[,3],y_pred_knn)

#-------- SVM Classifier
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')
y_pred_svm = predict(classifier, newdata = test_set[-3])

cm = table(test_set[,3],y_pred_svm)

#-------- Kernel SVM Classifier
library(e1071)
classifier_kSVM = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')
y_pred_svmK = predict(classifier_kSVM, newdata = test_set[-3])
cm = table(test_set[,3],y_pred_svmK)

#-------- Naive Bayes Classifier
library(e1071)
classifier_NB = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)
y_pred_NB = predict(classifier_NB, newdata = test_set[-3])
cm = table(test_set[,3],y_pred_NB)

#-------- DecisionTree Classifier
library(rpart)
classifier_DT = rpart(formula = Purchased ~ .,
                   data = training_set)
y_pred_DT = predict(classifier_DT,newdata= test_set[-3],type='class')
# here type='class' will result in y_pred to become vector.
# if type is not specified as 'class', y_pred will result in a matrix of probabilities for each class
cm = table(test_set[, 3], y_pred_DT)

#-------- RandomForest Classifier
library(randomForest)
set.seed(123)
classifier_RF = randomForest(x = training_set[-3],
                          y = training_set$Purchased,
                          ntree = 500)

y_pred_RF = predict(classifier_RF, newdata = test_set[-3])
cm = table(test_set[,3],y_pred_RF)