#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import pickle


# In[2]:


data = pd.read_csv('student-mat.csv',sep = ";")
data


# In[3]:


data["G3"].describe()


# In[4]:


data.shape


# In[5]:


data.isnull().any()


# In[6]:


#Duplicate value find and remove

def removingDuplicates(data):
    duplicateCount = data.duplicated().sum()
    print("Counts of Duplicates values: ", duplicateCount)  
    if duplicateCount >= 1:
        data.drop_duplicates(inplace=True)
        print('Duplicate values removing completed')
    else:
        print('No duplicate values found')
        
removingDuplicates(data)


# In[7]:


def dataVizual(data):
    plt.figure(figsize=(16, 9))
    dv = sns.countplot(data["G3"] )
    dv.axes.set_title("Final grade distribution",fontsize = 35)
    dv.set_xlabel("Final Grade",fontsize = 25)
    dv.set_ylabel('number of recodes' ,fontsize = 15)
    plt.show()
    
def classesVizualization(data):
    plt.figure(figsize=(16, 9))
    cv = sns.countplot(data["G3"],order=[i for i in range(21)])
    cv.axes.set_title("Final grade distribution with all classes",fontsize = 35)
    cv.set_xlabel("Final Grade classes",fontsize = 25)
    cv.set_ylabel('number of recodes' ,fontsize = 15)
    plt.show()
    
dataVizual(data)
classesVizualization(data)
    
    
    


# In[8]:


def Add_average_marks_to_data(data):
    data["GradeAvarage"] = (data["G1"] + data["G2"] + data["G3"]) /3
 
Add_average_marks_to_data(data)   
data.columns
data



# In[9]:


def gradeAllocation(data):
    All_Students_grades = []
    
    for aveMark in data["GradeAvarage"]:
        if aveMark >= (0.90 * data["GradeAvarage"].max()):
            All_Students_grades.append("A+")
        elif aveMark >= (0.80 * data["GradeAvarage"].max()):
            All_Students_grades.append("A")
        elif aveMark >= (0.75 * data["GradeAvarage"].max()):
            All_Students_grades.append("A-")
        elif aveMark >= (0.70 * data["GradeAvarage"].max()):
            All_Students_grades.append("B+")
        elif aveMark >= (0.65 * data["GradeAvarage"].max()):
            All_Students_grades.append("B")
        elif aveMark >= (0.60 * data["GradeAvarage"].max()):
            All_Students_grades.append("B-")
        elif aveMark >= (0.55 * data["GradeAvarage"].max()):
            All_Students_grades.append("C+")
        elif aveMark >= (0.45 * data["GradeAvarage"].max()):
            All_Students_grades.append("C")
        elif aveMark < (0.45 * data["GradeAvarage"].max()):
            All_Students_grades.append("F")
            
    data["StudentLevel"] = All_Students_grades
    return data
    
    


# In[10]:


PreprosesData = gradeAllocation(data)


# In[11]:


encoding = PreprosesData.drop("StudentLevel", axis=1)

object_cols = encoding.select_dtypes(include=[np.object])
print(object_cols.columns)


label_encoder = preprocessing.LabelEncoder()

# loop through every non numeric object
for col in object_cols:
    PreprosesData[col] = label_encoder.fit_transform(list(encoding[col]))


# In[12]:


PreprosesData.head()


# In[13]:


def read_in_and_split_data(data,label):
    X = data.drop(label, axis=1)
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# In[14]:


def classification_metrics(model, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))


# In[15]:


lable ='StudentLevel'
X_train, X_test, y_train, y_test = read_in_and_split_data(PreprosesData, lable)

# Train model

#Machinelearning_Algorithmn=RandomForestClassifier()
#model = Machinelearning_Algorithmn.fit(X_train, y_train)
#prediction = model.predict(X_test) # make predictions based on test data
#error = abs(prediction - y_test)

#print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
#print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")

#y_pred = model.predict(X_test)
#conf_matrix = confusion_matrix(y_test,y_pred)

pipeline = make_pipeline(StandardScaler(),  RandomForestClassifier())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test,y_pred)
classification_metrics(pipeline, conf_matrix)


# In[16]:


pickle.dump(model, open("student_perfomence.pkl", 'wb'))

