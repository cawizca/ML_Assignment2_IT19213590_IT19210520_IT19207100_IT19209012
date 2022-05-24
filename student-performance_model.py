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


data = pandas.read_csv('maths.csv', sep=";") 


# In[3]:


data.head(5)


# In[4]:


std_data = data.copy()


# In[5]:


def grade_calculation(mark):
    if mark >= 17:
        grade = 'A'
    elif mark >= 13:
        grade = 'B'
    elif mark >= 9:
        grade = 'C'
    else:
        grade = 'F'
    return grade

std_data["G1Grade"] = data["G1"].apply(lambda mark: grade_calculation(mark))
std_data["G2Grade"] = data["G2"].apply(lambda mark: grade_calculation(mark))
std_data["G3Grade"] = data["G3"].apply(lambda mark: grade_calculation(mark))


# In[6]:


std_data = std_data.drop(['G1'], axis=1)
std_data = std_data.drop(['G2'], axis=1)
std_data = std_data.drop(['G3'], axis=1)


# In[7]:


std_data.head()


# In[8]:


catergorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'romantic', 'G1Grade', 'G2Grade',
       'G3Grade']


# In[9]:


label_encoder = LabelEncoder()


# In[10]:


for col in catergorical_columns:
    std_data[col] = label_encoder.fit_transform(list(std_data[col]))


# In[11]:


std_data.head()


# In[12]:


X=std_data.drop(['G3Grade'],axis=1)
Y = std_data['G3Grade']


# In[13]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15, random_state=70)


# # Prediction With Linear Regression Model

# In[14]:


reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)


# In[15]:


LR_Predict = reg.predict(X_test)


# In[16]:


LR_score = reg.score(X_test, Y_test)


# In[17]:


LR_score


# In[18]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(LR_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Random Forest Model

# In[19]:


model_RF = RandomForestRegressor(n_estimators=1000)


# In[20]:


model_RF.fit(X_train, Y_train)


# In[21]:


RF_Predict = model_RF.predict(X_test)


# In[22]:


RF_score= model_RF.score(X_test, Y_test)


# In[23]:


RF_score


# In[24]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(RF_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Decision Tree Model

# In[25]:


model_DT = tree.DecisionTreeRegressor()


# In[26]:


model_DT.fit(X_train, Y_train)


# In[27]:


DT_Predict = model_DT.predict(X_test)


# In[28]:


DT_score= model_DT.score(X_test, Y_test)


# In[29]:


DT_score


# In[30]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(DT_Predict,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()


# # Prediction With Polynomial Regression Model

# In[ ]:





# In[31]:


pf = PolynomialFeatures(degree=3,include_bias=False)
pf


# In[32]:


X = pf.fit_transform(X)


# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15, random_state=51)


# In[34]:


model_PR = linear_model.LinearRegression()
model_PR.fit(X_train, Y_train)


# In[35]:


y_hat = model_PR.predict(X_test)


# In[36]:


PR_score= model_PR.score(X_test, Y_test)


# In[37]:


PR_score


# In[38]:


sns.distplot(Y_test,hist=False,color='r',label="Actucal Value")
sns.distplot(y_hat,hist=False,color='b',label="Fitted Value")

plt.title('Actual vs Fitted')
plt.xlabel('Score')
plt.ylabel('Grade')
plt.show()
plt.close()

