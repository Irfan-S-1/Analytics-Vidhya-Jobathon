#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.utils as sku
import sklearn.linear_model as sklm
import sklearn.neighbors as skn
import sklearn.ensemble as ske
import catboost as cb
import scipy.stats as sstats
import random
seed = 12
np.random.seed(seed)


# In[2]:


get_ipython().system('pip install pandas-profiling')
import pandas_profiling as pp


# In[3]:


#Import Train and Test datasets#

train=pd.read_csv("D:\\Data Science\\Project\\Jobathon Analytivs vidhya\\train_Df64byy.csv")
test=pd.read_csv("D:\\Data Science\\Project\\Jobathon Analytivs vidhya\\test_YCcRUnU.csv")


# In[69]:


targetfeature=train["Response"]


# In[5]:


train.shape,test.shape


# In[6]:


train.head()


# In[7]:


train.nunique()


# In[8]:


train.isnull().sum()/train.shape[0]*100


# In[9]:


test.isnull().sum()/test.shape[0]*100


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[12]:


train.dtypes


# In[13]:


train.corr()


# In[ ]:





# In[14]:


#Since Heath_indicator is a categorical feature and also it contains only 9 unique valueswe can treat missing values with mode#

train["Health Indicator"]=train["Health Indicator"].fillna(train["Health Indicator"].mode()[0])
test["Health Indicator"]=test["Health Indicator"].fillna(test["Health Indicator"].mode()[0])


# In[15]:


train["Health Indicator"].isnull().sum()


# In[16]:


#check target feature distribution
train["Response"].hist()
plt.show()


# In[242]:


sns.pairplot(train)
plt.show()


# In[91]:


# correlation heatmap for all features
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, annot=True)
plt.show()


# In[92]:


train.head()


# In[18]:


#Treating Holding_Policy_Duration#

#We can change 14+ years of Policy duration to 14#

train["Holding_Policy_Duration"].replace("14+","14",inplace=True)

test["Holding_Policy_Duration"].replace("14+","14",inplace=True)


# In[19]:


train.head()


# In[20]:


#Changing Datatype to float 

train["Holding_Policy_Duration"]=train["Holding_Policy_Duration"].astype(float)
test["Holding_Policy_Duration"]=test["Holding_Policy_Duration"].astype(float)


# In[21]:


#As there are no outliers hence we can treat missing values with mean#

train["Holding_Policy_Duration"]=train["Holding_Policy_Duration"].fillna(train["Holding_Policy_Duration"].mean())


test["Holding_Policy_Duration"]=test["Holding_Policy_Duration"].fillna(test["Holding_Policy_Duration"].mean())


# In[247]:


#Treating Holding_Policy_Type#

plt.figure(figsize=(8,5))
sns.boxplot("Holding_Policy_Type",data=train)


# In[22]:


#As there are no outliers hence we can treat missing values with mean#

train["Holding_Policy_Type"]=train["Holding_Policy_Type"].fillna(train["Holding_Policy_Type"].mean())


test["Holding_Policy_Type"]=test["Holding_Policy_Type"].fillna(test["Holding_Policy_Type"].mean())


# In[22]:


#Checking null values again#
train.isnull().sum()/train.shape[0]*100


# In[23]:


test.isnull().sum()/test.shape[0]*100


# In[24]:


train.corr()


# In[23]:


train.head()


# In[23]:


#Region code will not give much info hence we can drop it#

train.drop(["Region_Code"],axis=1,inplace=True)

test.drop(["Region_Code"],axis=1,inplace=True)


# In[24]:


#Count of Type of insurance customers are having#

plt.figure(figsize=(8,5))
sns.countplot('Reco_Insurance_Type',data=train,palette='ocean')


# In[174]:


plt.figure(figsize=(8,5))
sns.countplot("Accomodation_Type",data=train,palette='dark')


# In[175]:


plt.figure(figsize=(20,15))
sns.countplot("City_Code",data=train,palette='dark')


# In[176]:


plt.figure(figsize=(8,5))
sns.countplot("Health Indicator",data=train,palette='ocean')


# In[177]:


plt.figure(figsize=(10,8))
sns.barplot(y='Holding_Policy_Duration',x='Holding_Policy_Type',data=train,palette='flag')


# In[178]:


plt.figure(figsize=(12,10))
sns.barplot(y='Reco_Policy_Premium',x='Reco_Policy_Cat',data=train,palette='dark')


# In[26]:


Policy_wrt_Insurance=train.groupby("Is_Spouse")["Reco_Insurance_Type"].unique().value_counts()


# In[27]:


Policy_wrt_Insurance


# In[28]:


train.head()


# In[113]:


plt.figure(figsize=(15,12))
sns.barplot(y='Upper_Age',x='Lower_Age',data=train,palette='dark')


# In[114]:


profile = pp.ProfileReport(train, title='Pandas Profiling Report', explorative=True)
profile.to_file("profile.html")


# In[115]:


profile.to_notebook_iframe()


# In[116]:


train.head()


# In[117]:


train.corr()


# In[25]:


#Converting categorical to numerical data#

from sklearn.preprocessing import LabelEncoder


# In[26]:


lb=LabelEncoder()

train["City_Code"]=lb.fit_transform(train["City_Code"])
test["City_Code"]=lb.fit_transform(test["City_Code"])


# In[27]:


train["Accomodation_Type"]=lb.fit_transform(train["Accomodation_Type"])
test["Accomodation_Type"]=lb.fit_transform(test["Accomodation_Type"])


# In[28]:


train["Reco_Insurance_Type"]=lb.fit_transform(train["Reco_Insurance_Type"])
test["Reco_Insurance_Type"]=lb.fit_transform(test["Reco_Insurance_Type"])


# In[29]:


train["Is_Spouse"]=lb.fit_transform(train["Is_Spouse"])
test["Is_Spouse"]=lb.fit_transform(test["Is_Spouse"])


# In[30]:


train["Health Indicator"]=lb.fit_transform(train["Health Indicator"])
test["Health Indicator"]=lb.fit_transform(test["Health Indicator"])


# In[31]:


train.head()


# In[32]:


test.head()


# In[33]:


#Creating new columns#

#train["Age_diff"]=train["Upper_Age"]-train["Lower_Age"]
#test["Age_diff"]=test["Upper_Age"]-test["Lower_Age"]


# In[34]:


#Upper age and lower age columns are highly corelated hence will drop one feature#

train.drop(["Lower_Age"],axis=1)


# In[35]:


test.drop(["Lower_Age"],axis=1)


# In[36]:


def printScore(y_train, y_train_pred):
    print(skm.roc_auc_score(y_train, y_train_pred))


# In[37]:




df_y = targetfeature


# In[38]:


df_x=train.drop(["Response"],axis=1)


# In[39]:


pd.DataFrame(df_y)


# In[ ]:





# In[42]:


df_y


# In[45]:


X_train,X_test,Y_train,Y_test=train_test_split(df_x,df_y,test_size=0.3,random_state=seed)


# In[46]:


# scaler = skp.RobustScaler()
scaler = skp.MinMaxScaler()
#scaler = skp.StandardScaler()

# apply scaling to all numerical variables except dummy variables as they are already between 0 and 1
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# scale test data with transform()
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# view sample data
X_train.describe()


#X=pd.DataFrame(scaler.fit_transform(df_x),columns=df_x.columns)


# In[47]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[ ]:


#Model Building#


# In[48]:


classifier_NB = GaussianNB()

classifier_NB.fit(X_train, Y_train)

pred_NB_train=classifier_NB.predict(X_train)
print(np.mean(pred_NB_train==Y_train))
pred_NB_test=classifier_NB.predict(X_test)
print(np.mean(pred_NB_test==Y_test))

print(roc_auc_score(Y_train,pred_NB_train))
print(roc_auc_score(Y_test,pred_NB_test))


# In[49]:


classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_DT.fit(X_train,Y_train)

pred_DT_train=classifier_DT.predict(X_train)
print(np.mean(pred_DT_train==Y_train))
pred_DT_test=classifier_DT.predict(X_test)
print(np.mean(pred_DT_test==Y_test))

print(roc_auc_score(Y_train,pred_DT_train))
print(roc_auc_score(Y_test,pred_DT_test))


# In[50]:


classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,Y_train)

pred_LR_train=classifier_LR.predict(X_train)
print(np.mean(pred_LR_train==Y_train))
pred_LR_test=classifier_LR.predict(X_test)
print(np.mean(pred_LR_test==Y_test))

print(roc_auc_score(Y_train,pred_LR_train))
print(roc_auc_score(Y_test,pred_LR_test))


# In[51]:


classifier_ADA=AdaBoostClassifier()
classifier_ADA.fit(X_train,Y_train)

pred_ADA_train=classifier_ADA.predict(X_train)
print(np.mean(pred_ADA_train==Y_train))
pred_ADA_test=classifier_ADA.predict(X_test)
print(np.mean(pred_ADA_test==Y_test))

print(roc_auc_score(Y_train,pred_ADA_train))
print(roc_auc_score(Y_test,pred_ADA_test))


# In[52]:


classifier_RF=RandomForestClassifier()
classifier_RF.fit(X_train,Y_train)

pred_RF_train=classifier_RF.predict(X_train)
print(np.mean(pred_RF_train==Y_train))
pred_RF_test=classifier_RF.predict(X_test)
print(np.mean(pred_RF_test==Y_test))

print(roc_auc_score(Y_train,pred_RF_train))
print(roc_auc_score(Y_test,pred_RF_test))


# In[53]:


classifier_SVM=SVC()
classifier_SVM.fit(X_train,Y_train)

pred_SVM_train=classifier_SVM.predict(X_train)
print(np.mean(pred_SVM_train==Y_train))
pred_SVM_test=classifier_SVM.predict(X_test)
print(np.mean(pred_SVM_test==Y_test))


print(roc_auc_score(Y_train,pred_SVM_train))
print(roc_auc_score(Y_test,pred_SVM_test))


# In[41]:


classifier_KNN=KNeighborsClassifier()
classifier_KNN.fit(X_train,Y_train)

pred_KNN_train=classifier_KNN.predict(X_train)
print(np.mean(pred_KNN_train==Y_train))
pred_KNN_test=classifier_KNN.predict(X_test)
print(np.mean(pred_KNN_test==Y_test))

print(roc_auc_score(Y_train,pred_KNN_train))
print(roc_auc_score(Y_test,pred_KNN_test))


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier_KNN,df_x,df_y,cv=5)


# In[ ]:


#Hyper parameter tuning of SVM
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier_SVM,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)

grid=grid_search.fit(X_train,Y_train)
grid.best_params_


# In[42]:


classifier_KNN=KNeighborsClassifier()


# In[43]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier_KNN,df_x,df_y,cv=5)
score
score.mean() #accuracy=95.37


# In[56]:


import tensorflow as tf

print("TF version:-", tf.__version__)
import keras as k
tf.random.set_seed(seed)


# In[57]:


THRESHOLD = .999
bestModelPath = './best_model.hdf5'

class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > THRESHOLD):
            print("\n\nStopping training as we have reached our goal.")   
            self.model.stop_training = True

mycb = myCallback()
checkpoint = k.callbacks.ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True)

callbacks_list = [mycb,
                  checkpoint
                 ]
            
def plotHistory(history):
    print("Min. Validation ACC Score",min(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()


# In[59]:


epochs = 40

model_1 = k.models.Sequential([
    k.layers.Dense(2048, activation='relu', input_shape=(X_train.shape[1],)),
#     k.layers.Dropout(0.3),
    
    k.layers.Dense(1024, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(512, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(1, activation='sigmoid'),
])
print(model_1.summary())

model_1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
#                   tfa.metrics.F1Score(num_classes=1),
                  'accuracy'
              ]
)
history = model_1.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, 
                      batch_size=2048, )


# In[62]:


extra_model = ske.ExtraTreesClassifier(criterion='gini', random_state=1, verbose=0, n_jobs=-1,
                              n_estimators=200,max_depth=10,
                              min_samples_split = 5, min_samples_leaf = 1)
extra_model.fit(X_train, Y_train)


pred_extra_train=extra_model.predict(X_train)
print(np.mean(pred_extra_train==Y_train))
pred_extra_test=extra_model.predict(X_test)
print(np.mean(pred_extra_test==Y_test))

print(roc_auc_score(Y_train,pred_extra_train))
print(roc_auc_score(Y_test,pred_extra_test))


# In[65]:


import catboost as cb

cat_model = cb.CatBoostClassifier(loss_function='Logloss', verbose=0, eval_metric='AUC',
                           use_best_model=True, iterations=500)
cat_model.fit(X_train, Y_train, eval_set=(X_test, Y_test))
print(cat_model.best_score_)

y_train_pred = cat_model.predict(X_train)
y_test_pred = cat_model.predict(X_test)
print(skm.accuracy_score(Y_train, y_train_pred))
print(skm.accuracy_score(Y_test, y_test_pred))
printScore(Y_train, y_train_pred)
printScore(Y_test, y_test_pred)


# In[72]:


def getTestResults(m=None):
    df_final = train.sample(frac=1, random_state=1).reset_index(drop=True)
    test_cols = [x for x in df_final.columns if targetfeature not in x]
    df_final_test = df_test[test_cols]
    df_y = df_final.pop(targetFeature)
    df_X = df_final

#     scaler = skp.RobustScaler()
#     scaler = skp.MinMaxScaler()
    scaler = skp.StandardScaler()

    df_X = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)
    df_final_test = pd.DataFrame(scaler.transform(df_final_test), columns=df_X.columns)

    sample_weights = sku.class_weight.compute_sample_weight('balanced', df_y)
    
    if m is None:
        lmr = cb.CatBoostClassifier(loss_function='Logloss', verbose=0, eval_metric='AUC', class_weights=class_weights)
        lmr.fit(df_X, df_y)
        
        
    else:
        lmr = m

    # predict
    y_train_pred = lmr.predict(df_X)
    y_test_pred = lmr.predict(df_final_test)
    if m is not None:
        y_train_pred = [round(y[0]) for y in y_train_pred]
        y_test_pred = [round(y[0]) for y in y_test_pred]
    print(skm.accuracy_score(df_y, y_train_pred))
    printScore(df_y, y_train_pred)
    return y_test_pred

# ML models
results = getTestResults()


# In[ ]:


submission = pd.DataFrame({
    'ID': df_test['ID'],
    targetFeature: results,
})
print(submission.Response.value_counts())
submission.head()

