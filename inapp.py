import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

model_file = 'gradient_boosting_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

def predict_order_confirmation(data, model):
    # Ensure the input data matches the model's expected input
    # Create a DataFrame with all required columns (filling with dummy values if necessary)
    input_data = pd.DataFrame(data, index=[0], columns=[
        'Age', 'CreditScore', 'MonthlyIncome', 'Cost', 'Price', 'Quantity',
        'Gender', 'Country', 'State', 'City', 'Category', 'Product',
        'CampaignSchema ', 'PaymentMethod'
    ])
    
    input_data = input_data.fillna(0) 
    prediction = model.predict(input_data)[0]
    return prediction

def create_visualizations(df):
    pass

st.title('Order Confirmation Prediction')
st.markdown('This app predicts the order confirmation based on customer data.')

st.sidebar.header('Enter Customer Information')

age = st.sidebar.number_input('Age', min_value=18, max_value=100)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, max_value=1000)
monthly_income = st.sidebar.number_input('Monthly Income', min_value=0, max_value=100000)
category = st.sidebar.selectbox('Category', ['Category1', 'Category2'])  # Replace with actual categories from your dataset
product = st.sidebar.selectbox('Product', ['Product1', 'Product2'])     # Replace with actual products from your dataset
payment_method = st.sidebar.selectbox('Payment Method', ['Method1', 'Method2'])  # Replace with actual methods from your dataset
campaign_schema = st.sidebar.selectbox('Campaign Schema', ['Schema1', 'Schema2'])  # Replace with actual schemas from your dataset
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

input_data = {
    'Age': age,
    'CreditScore': credit_score,
    'MonthlyIncome': monthly_income,
    'Category': category,
    'Product': product,
    'PaymentMethod': payment_method,
    'CampaignSchema ': campaign_schema,  
    'Gender': gender,
    'Cost': 0,       
    'Price': 0,      
    'Quantity': 0,   
    'Country': 'Country',  
    'State': 'State',      
    'City': 'City',       
}

if st.sidebar.button('Predict'):
    prediction = predict_order_confirmation(input_data, model)
    confirmation = 'Confirmed' if prediction else 'Not Confirmed'
    st.sidebar.success(f'The predicted order confirmation is: {confirmation}')


df = pd.read_csv("https://raw.githubusercontent.com/kavyasri2099/insights_app/main/insights.csv")  


st.header('Data Visualizations')
create_visualizations(df)

