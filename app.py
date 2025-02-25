import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

## loading the model

dt=pickle.load(open('dt.pkl','rb'))
rf=pickle.load(open('rf.pkl','rb'))
knn=pickle.load(open('kn.pkl','rb'))
xgb=pickle.load(open('xg.pkl','rb'))
log=pickle.load(open('log.pkl','rb'))


# Title and description
st.markdown('<h1 style="color:blue;text-align:center;">Cancer Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown("""
Welcome to the Cancer Prediction Tool. This tool helps predict the likelihood of cancer based on various patient details. 
Please enter the necessary details in the sidebar, and click 'Predict' to see the results.
""")

st.sidebar.header('Enter the details of the patient')

Gender=st.sidebar.selectbox('Gender',('Male','Female'))

Age=st.sidebar.number_input('Age',min_value=18,max_value=90)

Tumor_Size=st.sidebar.number_input('Tumor Size',step=0.1,min_value=float(0.1),max_value=float(20))
Tumor_Grade=st.sidebar.selectbox('Tumor Grade',('High','Medium','Low'))

Symptoms_Severity=st.sidebar.selectbox('Symptoms Severity',('Mild','Moderate','Severe'))

Family_History=st.sidebar.selectbox('Family History',('Yes','No'))

Smoking_History=st.sidebar.selectbox('Smoking',('Former Smoker', 'Current Smoker', 'Non-Smoker'))

Alcohol_Consumption=st.sidebar.selectbox('Alcohol Consumption',('High', 'Moderate','Low'))

Exercise_Frequency=st.sidebar.selectbox('Exercise_frequency',('Regularly', 'Rarely', 'Occasionally', 'Never'))


if st.sidebar.button('Predict'):
    features=[[Age, Gender, Tumor_Size, Tumor_Grade, Symptoms_Severity,
       Family_History, Smoking_History, Alcohol_Consumption,
       Exercise_Frequency]]
    features=pd.DataFrame(features,columns=['Age', 'Gender', 'Tumor_Size', 'Tumor_Grade', 'Symptoms_Severity',
       'Family_History', 'Smoking_History', 'Alcohol_Consumption',
       'Exercise_Frequency'])
    
    pred=log.predict(features)
    if pred==1:
        st.header("Patient has :red[Cancer]",divider='rainbow')
    else:
        st.header("Patient has :green[no Cancer]",divider='rainbow')
    probability = norm.cdf(pred, loc=0, scale=1)
    plt.figure(figsize=(2,2))
    plt.pie([1-probability[0], probability[0]], labels=['No Cancer', 'Cancer'], autopct='%1.1f%%', shadow=True, startangle=140, explode=[0,0.05])
    plt.title('Probability of Cancer', fontsize=12)
    st.pyplot(plt)
