#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle
import warnings

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

school = 0 
sex =0
address=1
famsize = 0
Pstatus =1
Medu =4
Fedu = 2
Mjob =0
Fjob =3
reason=5
guardian=0
traveltime = 1
studytime =5
failures  =0
schoolsup =1
famsup=1
paid=1
activities =1
nursery=0
higher =0
internet=0
romantic=0
famrel=0
freetime=2
goout =2
Dalc =2
Walc =1
health =0
absences =1
G1 =20
G2 =20

feature_list = [school,sex,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2]
single_pred = np.array(feature_list).reshape(1,-1)
        
       
loaded_model = load_model('student_perfomence.pkl')
prediction = loaded_model.predict(single_pred)

print(prediction)

