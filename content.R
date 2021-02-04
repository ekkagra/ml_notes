# ---- Data Preprocessing ----
# most libraries already included. Just need to include some special libraries
# set working directory
dataset = read.csv('Data.csv')

# handling missing data
# subsituteMean
dataset$Age = ifelse(is.na(dataset$Age),
	ave(dataset$Age, FUN  = function(x) mean(x, na.rm = TRUE)),
	dataset$Age
	)

# categorical data encoding
# -- Note: factor output data is not numerical
dataset$Country = factor(dataset$Country,
						levels = c('France','Germany','Spain'),
						labels = c(1, 2, 3)
						)

# splitting dataset into Training and Test
install.packages('caTools')
library(caTools)
set.seed(123)
# sample by using one column
split = sample.split(datset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(datset,split == FALSE)

# Feature Scaling - only include numerical data
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
