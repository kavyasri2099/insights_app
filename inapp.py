import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pickled model
model_file = 'gradient_boosting_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to predict based on user inputs
def predict_order_confirmation(data, model):
    # Ensure the input data matches the model's expected input
    # Create a DataFrame with all required columns (filling with dummy values if necessary)
    input_data = pd.DataFrame(data, index=[0], columns=[
        'Age', 'CreditScore', 'MonthlyIncome', 'Cost', 'Price', 'Quantity',
        'Gender', 'Country', 'State', 'City', 'Category', 'Product',
        'CampaignSchema ', 'PaymentMethod'
    ])
    
    # Fill any missing values with default values
    input_data = input_data.fillna(0)  # Replace NaNs with zeros or appropriate defaults

    # Predict using the model
    prediction = model.predict(input_data)[0]
    return prediction

# Function to create visualizations
def create_visualizations(df):
    # Distribution of Age
    st.subheader('Distribution of Age')
    fig_age = plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    st.pyplot(fig_age)

    # Distribution of Credit Score
    st.subheader('Distribution of Credit Score')
    fig_credit = plt.figure(figsize=(8, 6))
    sns.histplot(df['CreditScore'], bins=20, kde=True)
    st.pyplot(fig_credit)

    # Monthly Income vs. Order Confirmation
    st.subheader('Monthly Income vs. Order Confirmation')
    fig_income = plt.figure(figsize=(8, 6))
    sns.boxplot(x='OrderConfirmation', y='MonthlyIncome', data=df)
    st.pyplot(fig_income)

    # Gender Distribution
    st.subheader('Gender Distribution')
    gender_counts = df['Gender'].value_counts()
    fig_gender = plt.figure(figsize=(8, 6))
    gender_counts.plot(kind='bar', rot=0)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    st.pyplot(fig_gender)

    # Category-wise Sales
    st.subheader('Category-wise Sales')
    category_sales = df.groupby('Category')['OrderConfirmation'].mean().sort_values(ascending=False)
    fig_category = plt.figure(figsize=(10, 6))
    category_sales.plot(kind='bar', rot=45)
    plt.title('Mean Order Confirmation by Category')
    plt.xlabel('Category')
    plt.ylabel('Mean Order Confirmation')
    st.pyplot(fig_category)

# Title and description for the Streamlit app
st.title('Order Confirmation Prediction')
st.markdown('This app predicts the order confirmation based on customer data.')

# Sidebar with input fields
st.sidebar.header('Enter Customer Information')

# Sample input fields based on provided features
age = st.sidebar.number_input('Age', min_value=18, max_value=100)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, max_value=1000)
monthly_income = st.sidebar.number_input('Monthly Income', min_value=0, max_value=100000)
category = st.sidebar.selectbox('Category', ['Category1', 'Category2'])  # Replace with actual categories from your dataset
product = st.sidebar.selectbox('Product', ['Product1', 'Product2'])     # Replace with actual products from your dataset
payment_method = st.sidebar.selectbox('Payment Method', ['Method1', 'Method2'])  # Replace with actual methods from your dataset
campaign_schema = st.sidebar.selectbox('Campaign Schema', ['Schema1', 'Schema2'])  # Replace with actual schemas from your dataset
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

# Create a dictionary to hold user input data
input_data = {
    'Age': age,
    'CreditScore': credit_score,
    'MonthlyIncome': monthly_income,
    'Category': category,
    'Product': product,
    'PaymentMethod': payment_method,
    'CampaignSchema ': campaign_schema,  # Note the space after CampaignSchema
    'Gender': gender,
    'Cost': 0,       # Dummy value, adjust as per your needs
    'Price': 0,      # Dummy value, adjust as per your needs
    'Quantity': 0,   # Dummy value, adjust as per your needs
    'Country': 'Country',  # Dummy value, adjust as per your needs
    'State': 'State',      # Dummy value, adjust as per your needs
    'City': 'City',        # Dummy value, adjust as per your needs
}

# Predict the order confirmation based on user inputs
if st.sidebar.button('Predict'):
    prediction = predict_order_confirmation(input_data, model)
    confirmation = 'Confirmed' if prediction else 'Not Confirmed'
    st.sidebar.success(f'The predicted order confirmation is: {confirmation}')

# Load your dataset for visualization (assuming df is your loaded DataFrame)
df = pd.read_csv("C:/Users/Lenovo/insights.csv")  # Adjust path as necessary

# Optionally, display all required visualizations
st.header('Data Visualizations')
create_visualizations(df)
