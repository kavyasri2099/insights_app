import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
model_file = 'gradient_boosting_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to predict based on user inputs
def predict_order_confirmation(data, model):
    # Ensure the input data matches the model's expected input
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)[0]
    return prediction

# Title and description for the Streamlit app
st.title('Order Confirmation Prediction')
st.markdown('This app predicts the order confirmation based on customer data.')

# Sidebar with input fields
st.sidebar.header('Enter Customer Information')

# Sample input fields (you can customize this based on your actual input fields)
age = st.sidebar.number_input('Age', min_value=18, max_value=100)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
monthly_income = st.sidebar.number_input('Monthly Income', min_value=0, max_value=100000)
# Add more input fields as per your features

# Create a dictionary to hold user input data
input_data = {
    'Age': age,
    'Gender': gender,
    'MonthlyIncome': monthly_income,
    # Add more fields here as per your features
}

# Predict the order confirmation based on user inputs
if st.sidebar.button('Predict'):
    prediction = predict_order_confirmation(input_data, model)
    st.sidebar.success(f'The predicted order confirmation is: {prediction}')

