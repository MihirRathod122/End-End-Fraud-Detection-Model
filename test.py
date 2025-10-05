import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

legit = data[data.Class==0]
fraud = data[data['Class']==1]

legit_sample = legit.sample(n=492,random_state=2)
data = pd.concat([legit_sample,fraud],axis=0)

X=data.drop('Class',axis=1)
y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)


model = LogisticRegression(max_iter=10000000)
model.fit(X_train,y_train)

train_acc = accuracy_score(model.predict(X_train),y_train)
test_acc = accuracy_score(model.predict(X_test),y_test)



#Web app
st.title("Credit Card Fraud Detection Model")  ###Title

input_df=st.text_input("Enter all required features")    ###User Input

input_df_splitted=input_df.split(',') ##whatever user input the user it should be , seperated.

Button=st.button("Submit") ##Submit button created
features=np.asarray(input_df_splitted,dtype=np.float64) ##machine understand input feature in from of array.

prediction=model.predict(features.reshape(1,-1))

if prediction[0]== 0:
    st.write("Legitimate Transaction")
else:
    st.write("Fraudlent Transaction")