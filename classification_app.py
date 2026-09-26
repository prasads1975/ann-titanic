import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")

pclass = st.slider('Passenger class',1,3)

sex = st.selectbox('Enter the passenger gender',['male','female'])

sibsp = st.slider('Enter the passenger total number of sibling and spouse',1,8)

parch = st.slider('Enter the passenger total number of parent and child',1,6)

fare = st.number_input('Enter the Fare for the Passenger')

embarked = st.selectbox('Enter the passenger onboarding port',['Southampton','Cheboug','Queenstown'])

data = pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

with open('classificiation_label_encoder.pkl','rb') as file:
    label = pickle.load(file)

with open('classificiation_one_hot_encoder.pkl','rb') as file:
    ohe = pickle.load(file)

with open('classificiation_scaler.pkl','rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

embarked = ohe.transform(data[['Embarked']])

embarked_df = pd.DataFrame(embarked,columns=ohe.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']),embarked_df],axis=1)

data[['Pclass','SibSp','Parch','Fare']] =  scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

model = load_model('classification_model.h5')

y = model.predict(data)

y = y[0][0]

def Chance(y):
    if y > 0.5:
        return 'Passenger will survive'
    else:
        return 'Passenger wont survive'

if st.button('Prdict'):
    st.write('Probability of survival :: ', y)
    st.write(Chance(y))