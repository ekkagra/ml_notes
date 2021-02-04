import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------- #
# --- Numpy Basic Manipulations --- #
# --------------------------------- #
df = pd.DataFrame([[1,2,3,4],[11,22,33,44]])
df.values
# Out[24]:
# array([[ 1,  2,  3,  4],
#        [11, 22, 33, 44]], dtype=int64)
df.values.ravel()
# Out[25]: array([ 1,  2,  3,  4, 11, 22, 33, 44], dtype=int64)
df.values.reshape(-1)
# Out[38]: array([ 1,  2,  3,  4, 11, 22, 33, 44], dtype=int64)
df.values.reshape(-1,1)
# Out[26]:
# array([[ 1],
#        [ 2],
#        [ 3],
#        [ 4],
#        [11],
#        [22],
#        [33],
#        [44]], dtype=int64)
df.values.reshape(1,-1)
# Out[28]: array([[ 1,  2,  3,  4, 11, 22, 33, 44]], dtype=int64)


# Dataset Load Here
dataset = pd.read_csv('C:/Users/ekksingh/ES/abc/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,2:3]

# --------------- #
# ----- EDA ----- #
# --------------- #
# Data Range Visualization
def percentile_grapher(dataset):
	''' Graphs the values at fixed percentiles for each numerical column.
	Helps to identify outlier data points. '''
    df = dataset.quantile(np.array(list(range(0,1001,5)))*0.001)
    ncol = len(df.columns)
    fig,axs = plt.subplots(ncol,figsize=(14,5*ncol))
    for i,col in enumerate(df.columns):
        print(i,col)
        axs[i].scatter(x=df[col].index,y=df[col])
        axs[i].set_title(col)

# Outlier Treatment: keep data within 1-99 percentile of 
def keep_within_percentile(dataset,lowerPtile=0.01,upperPtile=0.99):
    len1 = len(dataset)
    print(f'Initial: {len1}')
    percentiles = dataset.quantile([lowerPtile,upperPtile]).transpose().reset_index().values
    for ele in percentiles:
        dataset = dataset.loc[(dataset[ele[0]]>=ele[1]) & (dataset[ele[0]]<=ele[2])]
    print('Deleted:'+str(len1-len(dataset)))
    return dataset

# Correlation Visualization wrt a particular variable
def abs_corr_plot(dataset,target_col):
	''' Plot the absolute value of correlation sorted wrt a particular variable '''
    abs_cor = np.abs(dataset.corr())
    corr_sorted = abs_cor.sort_values(target_col,ascending=False).index.tolist()
    sns.heatmap(abs_cor[corr_sorted].sort_values(target_col,ascending=False),annot=True,cbar=False)

# Unique Value count finder
def unique_count(dataset,col_list):
	''' Prints out the unique values of columns mentioned in col_list '''
    count_list = []
    for col in col_list:
        count_list.append([col,dataset[col].nunique()])
    frame = pd.DataFrame(np.array(count_list),columns=['Col','Unique_counts'])
    frame['Unique_counts'] = frame['Unique_counts'].apply(pd.to_numeric)
    return frame.sort_values('Unique_counts',ascending=False) 

# Dummy variable generator
def generate_dummies(dataset,col_name):
    temp_dummies = pd.get_dummies(dataset[col_name],drop_first=True)
    new_col = [col_name+'_'+ ele for ele in list(temp_dummies.columns)]
    temp_dummies.columns = new_col
    return temp_dummies
def dummy_joiner(dataset,non_num_col_list):
    final_dummy = pd.DataFrame()
    for non_num_col in non_num_col_list:
        dummy_df = generate_dummies(dataset,non_num_col)
        if len(final_dummy) == 0:
            final_dummy = dummy_df
        else:
            final_dummy = final_dummy.join(dummy_df)
    return final_dummy

# ----------------------------- #
# ----- Linear Regression ----- #
# ----------------------------- #

# --- Preprocessing --- #
# Encode Categorical Data - two approaches
# 1. LabelEncoder -> OneHotEncoder
# 2. get_dummies

# -- 1. LabelEncoder -> OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X['Categorical_Column'] = labelencoder.fit_transform(X['Categorical_Column'])
onehotencoder = OneHotEncoder(categorical_features = [3])
# OneHotEncoder returns the binary encoded columns in the first positions of the special class
# This class is converted using toarray()
X = onehotencoder.fit_transform(X).toarray()
# Avoiding the Dummy Variable Trap (DONOT consider all dummy variable. n-1 columns to be considered)
X = pd.DataFrame(X[:, 1:])

# -- 2. get_dummies
dummy = pd.get_dummmies(X['Categorical_Column'])
dummy = dummy.iloc[:,1:]
X = X.join(dummy)

# --- Extras --- #
# Missing value Treatment
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# train_test_split randomly splits the data and original sequence order is shuffled
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    random_state=33,
                                                    test_size = 0.2)

# -------------------------------- #
# --- Simple Linear Regression --- #
# -------------------------------- #
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
# Coefficients of columns
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
# Evaluation
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))

# ------------------------------------ #
# --- Polynomial Linear Regression --- #
# ------------------------------------ #
from sklearn.preprocessing import PolynomialFeatures
# PolynomialFeatures calculates all combination of powers <= degree
# eg. when two columns are give say a and b with degree=3, then columns generated are:
# 1, a, b, a^2, ab, b^2, a^3, (a^2)(b ), (a)(b^2), b^3
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)
lm_poly = LinearRegression()
lm_poly.fit(X_poly,y)
y_pred_poly = lm_poly.predict(X_test)

# ----------- # 
# --- SVR --- # 
# ----------- #
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X,y)
y_pred = svr_regressor.predict([[6.5]])
y_pred = sc_y.inverse_transform(y_pred)

# ------------------------------- #
# --- Decision Tree Regressor --- #
# ------------------------------- #
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X,y)
y_pred = dt_regressor.predict(X_test)

# ------------------------------- #
# --- Random Forest Regressor --- #
# ------------------------------- #
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=10,
                                     random_state=0)
rf_regressor.fit(X,y)
y_pred = rf_regressor.predict(X_test)

# --------------------------------------- #
# --- Evaluation of Regression models --- #
# --------------------------------------- #
'''
1. R_square => [1 - SS(res)/SS(total)]
    where SS(res) => Residual sum of squares - Error between model and true values
          SS(tot) => Total sum of squares - Error between average line and true values
Value lies between (-inf, 1)
Measure of GOOD FIT. How much better the model fits as compared to an average line.
Problem:
    Increasing the number of independant variable always increases R_square.
    Solution: -> Consider Adjusted R_square

2. Adjusted R_square
    Measures goodness of fit while introducing a penalty for increasing the number of 
    independant variables.
'''

# --- MODEL SELECTION PROCESS --- #
'''
1. Find whether the model is linear or non-linear
2. For linear cases, go for Simple Linear Regression if single variable is there
else multiple linear regression if many variables are present
3. For non-linear cases, go with either 
Polynomial
SVR
DecisionTree
RandomForest
'''

# ---------------------- #
# --- Classification --- #
# ---------------------- #

# --- Logistic Regression --- #
# --------------------------- #
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Evaluation Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# --- KNN Classifier --- #
# ---------------------- #
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# --- SVM Classifier --- #
# ---------------------- #
'''
SVM - Support Vector Machines - use vectors which are obtained from the training data points
which are closest to the supposed decision boundary to obtain the decision hyperplane

Support vectors - vectors joining hyperplane and nearest data points

This algorithm looks at the most extreme boundary data points to find the decision boundary
Goal is to maximize the sum of support vector distances from hyperplane
'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# --- Kernel SVM Classifier --- #
# ----------------------------- #
'''
Kernel SVM Classifier projects data using some function to higher dimension
In this higher dimension, data can be linearly separated.
'''
# >>> Feature scaling required - Yes
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

# --- Naive Bayes classifier --- #
# ------------------------------ #
'''
Classifier based on calculating probabilities for various classes given data point and assigning class with maximum probability.
Assume space around point whose class is to be predicted as 'X'
Then calculate using Bayes Theorem
P(class1|X) = P(X|class1)*P(class1)/P(X)  [Prob. of class1 given X]
P(class2|X) = P(X|class2)*P(class2)/P(X)  [Prob. of class2 given X]

Assign class which has higher probability
decision made from P(class1|X) > P(class2|X)
'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# --- Decision Tree Classfier --- #
# ------------------------------- #
"""
Make split such as to increase the overall entropy
Very simple and not very useful on their own, but other algorithms based on Decision Tree have proved quite effective.
In sample space, make splits such as to differentiate classes as much as possbile.
Doesnot depend on Euclidean distance
"""
from sklearn.tree import DecisionTreeClassifier
# choose criterion as entropy which measures the quality of split
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.method(X_train,y_train)
y_pred = classifier.predict(X_test)

# --- Random Forest Classifier --- #
# -------------------------------- #
"""
Ensemble Learning - Algorithms which use many algorithms to make predictions/classifications
- RF use many DecisionTrees which get fitted on subsets of training data.
- Final class is assigned which has a majority of vote among all the DecisionTrees
"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,
                                    criterion='entropy',
                                    random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

