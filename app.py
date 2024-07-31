import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model and scaler
model = pickle.load(open('../models/model.pkl', 'rb'))
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))

def predict(features):
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction

st.title('Menopause Risk Prediction')
st.write("Enter your details to predict the risk of delayed or early menopause:")

age = st.number_input('Age', min_value=18, max_value=100)
bmi = st.number_input('BMI', min_value=10, max_value=50)
# Add more input fields as needed

if st.button('Predict'):
    features = pd.DataFrame([[age, bmi]], columns=['age', 'BMI'])
    prediction = predict(features)
    st.write(f'Prediction: {prediction[0]}')
