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


# Read the dataset 
data = pd.read_csv('student-mat.csv',sep = ";")
data


# In[3]:


# copy the dataset into a new variable
std_data = data.copy()


# In[4]:


#Display the stastistical summary
data["G3"].describe()


# In[5]:


# Display the dimensions of datset
data.shape


# In[6]:


# Find null values in dataset
data.isnull().any()


# In[7]:


# Find duplicate values and remove

def removingDuplicates(data):
    duplicateCount = data.duplicated().sum()
    print("Counts of Duplicates values: ", duplicateCount)  
    if duplicateCount >= 1:
        data.drop_duplicates(inplace=True)
        print('Duplicate values removing completed')
    else:
        print('No duplicate values found')
        
removingDuplicates(data)


# In[8]:


# Data visualize using count plot
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
    
    
    


# In[9]:


#Calculate average marks and assign into new column
def Add_average_marks_to_data(data):
    data["GradeAvarage"] = (data["G1"] + data["G2"] + data["G3"]) /3
 
Add_average_marks_to_data(data)   
data.columns
data



# In[10]:


#Grade classification according to average mark
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
    
    


# In[11]:


PreprosesData = gradeAllocation(data)


# In[12]:


#Drop StudentLevel column from encoding
encoding = PreprosesData.drop("StudentLevel", axis=1)

#Get all non numeric columns
object_cols = encoding.select_dtypes(include=[np.object])
print(object_cols.columns)


label_encoder = preprocessing.LabelEncoder()

# Encode all non numeric column loop through
for col in object_cols:
    PreprosesData[col] = label_encoder.fit_transform(list(encoding[col]))


# In[13]:


# Display the head of output
PreprosesData.head()


# In[14]:


# Split dataset for testing and training.
def read_in_and_split_data(data,label):
    X = data.drop(label, axis=1)
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# In[15]:


# defining the grade calculation method
def calc_grade(mark):
    if mark >= 17:
        grading = 'A'
    elif mark >= 13:
        grading = 'B'
    elif mark >= 9:
        grading = 'C'
    else:
        grading = 'F'
    return grading

# assigning grades into a new column
std_data["G1Grade"] = data["G1"].apply(lambda mark: calc_grade(mark))
std_data["G2Grade"] = data["G2"].apply(lambda mark: calc_grade(mark))
std_data["G3Grade"] = data["G3"].apply(lambda mark: calc_grade(mark))


# In[16]:


# removing the old attributes
std_data = std_data.drop(['G1'], axis=1)
std_data = std_data.drop(['G2'], axis=1)
std_data = std_data.drop(['G3'], axis=1)


# In[17]:


# display the head of the modified dataset
std_data.head()


# In[18]:


# plot the correlation of the datatset
plt.figure(figsize=(20, 20))
sns.heatmap(std_data.corr().round(2), annot=True)


# In[19]:


# Calculate and plot the model acuracy
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


# In[20]:


lable ='StudentLevel'
X_train, X_test, y_train, y_test = read_in_and_split_data(PreprosesData, lable)

# Implement the Random Forest Classifier algorithem
pipeline = make_pipeline(StandardScaler(),  RandomForestClassifier())
model = pipeline.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test,y_pred)
classification_metrics(pipeline, conf_matrix)


# In[21]:


# Export the trained model as pickle file
pickle.dump(model, open("student_perfomence.pkl", 'wb'))

