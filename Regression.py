#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import json
import copy
# In[3]:
dataset_orig = pd.read_csv(r'C:\Users\ekksingh\ES\abc\dataset\melb_data.csv\melb_data.csv')
# In[4]:
dataset = dataset_orig.copy()
# In[5]:
dataset.head()
# In[6]:
dataset.info()
# In[6]:
dataset.describe().transpose()
# In[7]:
def percentile_grapher(dataset):
    df = dataset.quantile(np.array(list(range(0,1001,5)))*0.001)
    ncol = len(df.columns)
    fig,axs = plt.subplots(ncol,figsize=(14,5*ncol))
    for i,col in enumerate(df.columns):
        print(i,col)
        axs[i].scatter(x=df[col].index,y=df[col])
        axs[i].set_title(col)
# In[10]:
sns.pairplot(dataset)
# In[10]:
percentile_grapher(dataset)
# ## Excluding data at extreme percentiles(<=1,>=99) for each column
# In[63]:
len1 = len(dataset)
for ele in dataset.quantile([0.01,0.99]).transpose().reset_index().values:
    dataset = dataset.loc[(dataset[ele[0]]>=ele[1]) & (dataset[ele[0]]<=ele[2])]
print('Deleted:'+str(len1-len(dataset)))
# ### Outliers candidates - Percentile Based
# Rooms
# 
# Price **
# 
# Distance
# 
# Bedroom2
# 
# Bathroom
# 
# Car
# 
# Landsize **
# 
# BuildingArea **
# 
# YearBuilt
# 
# In[9]:
plt.figure(figsize=(7,7))
sns.scatterplot(x='Longtitude',y='Lattitude',data=dataset,hue='Price',alpha=0.3,palette='RdYlGn' ,edgecolor=None)
# In[19]:
plt.figure(figsize=(18,5))
sns.boxplot(dataset['Price'],fliersize=10,whis=1.5)
# ### Finding Outliers - Range, Std based
# In[20]:
((dataset.describe().loc['max']-dataset.describe().loc['min'])/dataset.describe().loc['mean']*100).sort_values(ascending=False)
# In[21]:
((dataset.describe().loc['std'])/dataset.describe().loc['mean']*100).sort_values(ascending=False)
# In[ ]:
plt.figure(figsize=(14,8))
sns.boxplot(dataset.loc[dataset['Landsize']<1347]['Landsize'])
print(len(dataset.loc[dataset['Landsize']>1347]))
# In[ ]:
plt.figure(figsize=(14,8))
sns.boxplot(dataset.loc[dataset['BuildingArea']<263]['BuildingArea'])
print(len(dataset.loc[dataset['BuildingArea']>263]))
# ## Missing data plot
# In[66]:
plt.figure(figsize=(14,8))
sns.heatmap(pd.isnull(dataset))
# ## Dataset Cleaning
# In[ ]:
dataset.drop(dataset[dataset['Price']>3000000].index,inplace=True)
# In[65]:
dataset.dropna(subset=['Car'],inplace=True)
dataset.dropna(subset=['CouncilArea'],inplace=True)
# In[ ]:
dataset.drop(dataset[dataset['YearBuilt']<1800].index,inplace=True)
# In[ ]:
dataset.drop(dataset[dataset['BuildingArea']>263].index,inplace=True)
# In[ ]:
dataset.drop(dataset[dataset['Landsize']>1340].index,inplace=True)
# ### Correlation Absolute Values Heatmap
# In[7]:
def abs_corr_plot(dataset,target_col):
    abs_cor = np.abs(dataset.corr())
    corr_sorted = abs_cor.sort_values(target_col,ascending=False).index.tolist()
    sns.heatmap(abs_cor[corr_sorted].sort_values(target_col,ascending=False),annot=True,cbar=False)
# In[8]:
plt.figure(figsize=(14,8))
abs_corr_plot(dataset,'Price')
# In[27]:
plt.figure(figsize=(14,8))
abs_corr_plot(dataset,'YearBuilt')
# ### Column Unique Count Finder
# In[28]:
def unique_count(dataset,col_list):
    count_list = []
    for col in col_list:
#         print('{0:15}:{1}'.format(col,dataset[col].nunique()))
        count_list.append([col,dataset[col].nunique()])
    frame = pd.DataFrame(np.array(count_list),columns=['Col','Unique_counts'])
    frame['Unique_counts'] = frame['Unique_counts'].apply(pd.to_numeric)
    return frame.sort_values('Unique_counts',ascending=False)
# In[ ]:
unique_count(dataset,list(dataset.columns))
# In[29]:
non_numeric_col = list(set(dataset.columns)-set(dataset.corr().columns))
unique_count(dataset,non_numeric_col)
# In[30]:
plt.figure(figsize=(10,9))
sns.countplot(y=dataset['Method'],)
# ### Dummy Variable Generator
# In[18]:
def generate_dummies(dataset,col_name):
    temp_dummies = pd.get_dummies(dataset[col_name],drop_first=True)
    new_col = [col_name+'_'+ ele for ele in list(temp_dummies.columns)]
    temp_dummies.columns = new_col
    return temp_dummies
# In[ ]:
dummy_df = generate_dummies(dataset,'Type')
# ### Dummy Joiner
# In[19]:
def dummy_joiner(dataset,non_num_col_list):
    final_dummy = pd.DataFrame()
    for non_num_col in non_num_col_list:
        dummy_df = generate_dummies(dataset,non_num_col)
        if len(final_dummy) == 0:
            final_dummy = dummy_df
        else:
            final_dummy = final_dummy.join(dummy_df)
    return final_dummy
# In[12]:
full_dummy = dummy_joiner(dataset,['Regionname','Method','Type','CouncilArea'])
# In[13]:
plt.figure(figsize=(40,30))
abs_corr_plot(dataset.join(full_dummy),'Price')
# BA ~ Rooms + Bathroom + Type + Landsize + Car + Distance
# In[ ]:
df_BA = dataset.dropna(subset=['BuildingArea'])
df_BAna = dataset[pd.isnull(dataset['BuildingArea'])]
print(len(df_BA),len(df_BAna))
# In[ ]:
cols_ba = ['Rooms','Type','Distance','Bathroom','Landsize','Car']
Xba = df_BA[cols_ba]
yba = df_BA['BuildingArea']
Xbana = df_BAna[cols_ba]
ybana = df_BAna['BuildingArea']
# In[ ]:
dummy_Xba = generate_dummies(Xba,'Type')
Xba = Xba.join(dummy_Xba)
dummy_Xbana = generate_dummies(Xbana,'Type')
Xbana = Xbana.join(dummy_Xbana)
Xbana.drop('Type',axis=1,inplace=True)
# In[ ]:
Xba.drop('Type',axis=1,inplace=True)
# In[ ]:
Xba.head()
# In[ ]:
lm_regressor = LinearRegression()
svr_regressor = SVR(kernel='linear',C=10)
dt_regressor = DecisionTreeRegressor()
rf_regressor = RandomForestRegressor()
eval_metrics_ba,mdl_ba = fit_predict(Xba,yba,rf_regressor,10)
# In[ ]:
resultsba = pd.DataFrame(data=np.array(eval_metrics_ba),columns=['i','mse','mae','mpe','r2'])
resultsba
# In[ ]:
y_pred_baNa = mdl_ba.predict(Xbana)
Xbana['BuildingArea'] = y_pred_baNa
# In[ ]:
dataset.loc[pd.isnull(dataset['BuildingArea']),'BuildingArea'] = Xbana['BuildingArea']
# In[ ]:
full_dummy.columns
# PRICE => 'Room','Bathroom','Type_h','Regionname_Southern Metropolitan','YearBuilt','Car','CouncilArea_Boroondara'
# 
# BA => 'Room', 'Type_u'
# 
# YB => 'Type
# In[ ]:
sns.distplot(dataset['YearBuilt'],bins=50)
# In[ ]:
print(dataset['Distance'].describe())
sns.distplot(dataset['Distance'],bins=20)
# In[ ]:
dataset['Distance_cat'] = pd.cut(dataset['Distance'],10,labels=False)
# In[ ]:
plt.figure(figsize=(15,6))
# sns.distplot(dataset['YearBuilt'],bins=100)
# sns.scatterplot(x='YearBuilt',y='Price',data=dataset,hue='Type')
# sns.boxenplot(x='Distance',y='YearBuilt',data=dataset)
# sns.boxenplot(x='Distance_cat',y='YearBuilt',data=dataset)
# sns.boxenplot(x='Type',y='YearBuilt',data=dataset)
# sns.boxenplot(x='Type',y='BuildingArea',data=dataset.drop(dataset[dataset['BuildingArea']>800].index))
# sns.scatterplot(x='Distance_cat',y='YearBuilt',data=dataset,hue='Type')
# In[ ]:
na_index = dataset.index.isin(dataset[pd.isna(dataset['YearBuilt'])].index)
year_NA = dataset[na_index]
year_valid = dataset[~na_index]
# In[ ]:
y1 = year_valid[['Distance_cat','Type','YearBuilt','BuildingArea']]
# In[ ]:
y1.head()
# In[ ]:
pv1 = pd.pivot_table(data=y1,index=['Type'],columns='Distance_cat',values='YearBuilt',aggfunc=np.median)
pv1 = pv1.fillna(pv1.mean())
# In[ ]:
pv1
# In[ ]:
typeDist_Year_json= json.loads(pv1.to_json())
# In[ ]:
dataset['YearBuilt_1']=dataset['YearBuilt']
dataset.loc[dataset['YearBuilt'].isnull(),['YearBuilt_1']]=dataset.apply(lambda row: round(typeDist_Year_json[str(row['Distance_cat'])][row['Type']]),axis=1)
dataset[['YearBuilt','YearBuilt_1','Distance_cat','Type']]
#  # FINAL
# In[242]:
# d1 = dataset.loc[(dataset['Price']<3500000) & (dataset['Price']>250000)]
d1 = dataset
print(d1.columns)
# In[243]:
d1.iloc[0]
# In[244]:
# Selecting non-object columns
dataset.select_dtypes(exclude=['object']).columns
# In[245]:
dataset_cols = []
# dataset_cols =['Rooms','Bathroom','YearBuilt_1','Car']
dataset_cols = ['Rooms', 'Distance', 'Postcode', 'Bathroom', 'Car',
       'Landsize', 'Lattitude', 'Longtitude',
       'Propertycount']
dummy_gen_cols = ['Type','Regionname','CouncilArea']
dummy_cols = ['Type_h','Regionname_Southern Metropolitan','CouncilArea_Boroondara']
final_cols = dataset_cols+ dummy_gen_cols
print(final_cols)
# In[246]:
X = d1[final_cols]
# In[247]:
y = d1['Price']
y
# In[248]:
dummy_x = dummy_joiner(X,dummy_gen_cols)
# relevant_cols = dummy_x[dummy_cols]
relevant_cols = dummy_x
relevant_cols
# In[249]:
X= X.join(relevant_cols)
X = X.drop(dummy_gen_cols,axis=1)
# In[27]:
X.iloc[0]
# In[205]:
plt.figure(figsize=(12,4))
sns.boxenplot(y)
# In[250]:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
# In[149]:
X = pd.DataFrame(sc_X.fit_transform(X),columns=X.columns)
# y = pd.DataFrame(sc_Y.fit_transform(pd.DataFrame(y)))
# In[39]:
sc_Y.fit(y.values.reshape(-1,1))
# In[41]:
y_trans = sc_Y.transform(y.values.reshape(-1,1))
# In[42]:
y_trans
# In[52]:
sc_Y.inverse_transform(y_trans)
# In[53]:
y
# In[207]:
plt.figure(figsize=(18,6))
print(y.describe())
sns.distplot(y)
# In[251]:
from sklearn.model_selection import train_test_split
# In[252]:
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import explained_variance_score
# In[253]:
def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# ## Fit Predict Model
# In[116]:
def fit_predict(X,y,model,iterations=1):
    eval_metrics = []
    for i in range(0,iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
#         minmax = MinMaxScaler()
#         minmax.fit(X_train)
#         X_train = minmax.transform(X_train)
#         X_test = minmax.transform(X_test)
        sc_X = StandardScaler()
        sc_X.fit(X_train)
        X_train = sc_X.transform(X_train)
        X_test = sc_X.transform(X_test)
        mdl = model
        mdl.fit(X_train,y_train.values.ravel())
        y_pred = pd.DataFrame(mdl.predict(X_test))
        print(f'Finished Run: {i}')
        rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
        mae = metrics.mean_absolute_error(y_test,y_pred)
        mpe = mean_absolute_percentage_error(y_test.values,y_pred.values)
        r2 = metrics.r2_score(y_test,y_pred)
        evs = explained_variance_score(y_test,y_pred)
        eval_metrics.append([rmse,mae,mpe,r2,evs])
    return eval_metrics,mdl
# In[115]:
models = {
#     "lin_reg": LinearRegression(),
#     "svr_1": SVR(kernel='linear'),
#     "svr_2": SVR(kernel='poly'),
#     "dt_reg": DecisionTreeRegressor(),
    "rf_reg": RandomForestRegressor()
}
# In[76]:
full_results = pd.DataFrame(columns=['model','rmse','mae','mpe','r2','evs'])
for modl in models:
    eval_metrics,mdl = fit_predict(X,y,models[modl],10)
    result = pd.DataFrame(data=np.array(eval_metrics),columns=['rmse','mae','mpe','r2','evs'])
    print(result.mean())
    result = pd.DataFrame(result.mean()).transpose()
    result['model'] = modl
    result = result[['model','rmse','mae','mpe','r2','evs']]
    full_results=pd.concat([full_results,result])
# In[ ]:
# X_train
# pd.DataFrame({'Feature':list(X_train.columns),'Coef':list(lm.coef_)})
# In[77]:
# results = pd.DataFrame(data=np.array(em),columns=['i','rmse','mae','mpe','r2','evs'])
# results
full_results
# In[259]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
mdl.fit(X_train,y_train)
y_pred = pd.DataFrame(mdl.predict(X_test))
plt.figure(figsize=(15,6))
sns.boxplot(y_pred - y_test)
# In[213]:
residual_quantiles = (y_pred-y_test).quantile(np.array(list(range(0,101,2)))*0.01)
plt.scatter(x=residual_quantiles.index,y=residual_quantiles.values)
# ## Tensorflow
# In[307]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train.shape
# In[295]:
minmax2 = MinMaxScaler()
minmax2.fit(X_train)
X_train = minmax2.transform(X_train)
X_test = minmax2.transform(X_test)
# In[256]:
sns.distplot(y)
# In[102]:
minmx = MinMaxScaler()
minmx.fit(y.values.reshape(-1,1))
y_mn = minmx.transform(y.values.reshape(-1,1))
# In[112]:
scy = StandardScaler()
scy.fit(y.values.reshape(-1,1))
y_sc = scy.transform(y.values.reshape(-1,1))
# In[113]:
sns.distplot(y)
# In[108]:
sns.distplot(y_mn)
# In[114]:
sns.distplot(y_sc)
# In[237]:
from tensorflow.keras import Sequential
# In[238]:
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
# In[296]:
tf_model = Sequential()
tf_model.add(Dense(45,activation='relu'))
# tf_model.add(Dropout(rate=0.2))
tf_model.add(Dense(22,activation='relu'))
# tf_model.add(Dropout(rate=0.2))
tf_model.add(Dense(11,activation='relu'))
# tf_model.add(Dropout(rate=0.2))
tf_model.add(Dense(5,activation='relu'))
# tf_model.add(Dropout(rate=0.2))
tf_model.add(Dense(1))
tf_model.compile(optimizer='adam',loss='mse')
# In[297]:
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=30)
# In[298]:
tf_model.fit(x=X_train,y=y_train.values,validation_data=(X_test,y_test.values),
            epochs=500,batch_size=256,callbacks=[early_stop])
# In[299]:
losses = pd.DataFrame(tf_model.history.history)
losses.plot()
# In[300]:
y_pred_tf = tf_model.predict(X_test)
# In[301]:
metrics.explained_variance_score(y_test.values,y_pred_tf.ravel())
# In[302]:
np.sqrt(metrics.mean_squared_error(y_test,y_pred_tf.ravel()))
# In[223]:
# y_test2 = sc_Y.inverse_transform(y_test)
# y_pred_tf2 = sc_Y.inverse_transform(y_pred_tf)
# In[303]:
mean_absolute_percentage_error(y_test.values,y_pred_tf.ravel())
# In[304]:
x=pd.DataFrame(np.array([[1,2],[2,3],[3,5],[4,7],[5,9]]))
x
# In[305]:
y=pd.DataFrame(np.array([2,3,4,5,6]))
y
# In[306]:
x_tr,x_te,y_tr,y_te = train_test_split(x,y)
# In[217]:
x_tr
# In[218]:
y_tr
# In[235]:
x_tr.values
# In[221]:
np.array([[2,3],[3,4],[4,5]]).reshape(-1,1)
