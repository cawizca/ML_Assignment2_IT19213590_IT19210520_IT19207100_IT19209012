#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


# In[2]:


pd = pandas.read_csv('maths.csv', sep=";") 


# In[3]:


pd.isnull().any()


# In[4]:


pd = pd.dropna(how='any',axis=0) 


# In[5]:


pd.head(5)


# In[6]:


Performance_data = pd.copy()


# In[7]:


def grade_calculation(value):
    if value >= 16:
        grade = 'A'
    elif value >= 12:
        grade = 'B'
    elif value >= 8:
        grade = 'C'
    else:
        grade = 'F'
    return grade

Performance_data["Grade_G1"] = pd["G1"].apply(lambda value: grade_calculation(value))
Performance_data["Grade_G2"] = pd["G2"].apply(lambda value: grade_calculation(value))
Performance_data["Grade_G3"] = pd["G3"].apply(lambda value: grade_calculation(value))


# In[8]:


Performance_data = Performance_data.drop(['G1','G2','G3'], axis=1)


# In[9]:


Performance_data.head()


# In[10]:


catergorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic', 'Grade_G1', 'Grade_G2',
       'Grade_G3']


# In[11]:


label_encoder = LabelEncoder()


# In[12]:


for col in catergorical_columns:
    Performance_data[col] = label_encoder.fit_transform(list(Performance_data[col]))


# In[13]:


Performance_data.head()


# In[14]:


X=Performance_data.drop(['Grade_G3'],axis=1)
Y = Performance_data['Grade_G3']


# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15, random_state=70)


# # Prediction With Linear Regression Model

# In[16]:


reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)


# In[17]:


LR_Predict = reg.predict(X_test)


# In[18]:


LR_score = reg.score(X_test, Y_test)


# In[19]:


LR_score


# In[20]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(LR_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Random Forest Model

# In[21]:


model_RF = RandomForestRegressor(n_estimators=1000)


# In[22]:


model_RF.fit(X_train, Y_train)


# In[23]:


RF_Predict = model_RF.predict(X_test)


# In[24]:


RF_score= model_RF.score(X_test, Y_test)


# In[25]:


RF_score


# In[26]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(RF_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Decision Tree Model

# In[27]:


model_DT = tree.DecisionTreeRegressor()


# In[28]:


model_DT.fit(X_train, Y_train)


# In[29]:


DT_Predict = model_DT.predict(X_test)


# In[30]:


DT_score= model_DT.score(X_test, Y_test)


# In[31]:


DT_score


# In[32]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(DT_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Polynomial Regression Model

# In[ ]:





# In[33]:


pf = PolynomialFeatures(degree=3,include_bias=False)
pf


# In[34]:


X = pf.fit_transform(X)


# In[35]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15, random_state=51)


# In[36]:


model_PR = linear_model.LinearRegression()
model_PR.fit(X_train, Y_train)


# In[37]:


y_hat = model_PR.predict(X_test)


# In[38]:


PR_score= model_PR.score(X_test, Y_test)


# In[39]:


PR_score


# In[40]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(y_hat,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# In[41]:


print("Multiple Linear Regression Model Score is ", round(LR_score*100))
print("Decision tree  Regression Model Score is ",round(DT_score*100))
print("Random Forest Regression Model Score is ",round(RF_score*100))
print("Polynomial Regression Model Score is ",round(PR_score*100))

models_score =pandas.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random forest Regression'],
                            'Score':[LR_score,DT_score,RF_score]
                           })
models_score.sort_values(by='Score',ascending=False)


# In[42]:


import pickle
filename='student_model.pkl'
pickle.dump(model_RF, open(filename, 'wb'))

