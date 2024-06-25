import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
model_file = 'gradient_boosting_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to predict based on user inputs
def predict_order_confirmation(data, model):
    # Create a DataFrame with all required columns
    input_data = pd.DataFrame(data, index=[0])
    
    # Predict using the model
    prediction = model.predict(input_data)[0]
    return prediction

# Title and description for the Streamlit app
st.title('Order Confirmation Prediction')
st.markdown('This app predicts the order confirmation based on customer data.')

# Sidebar with input fields
st.sidebar.header('Enter Customer Information')

# Sample input fields based on provided features
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, max_value=1000, value=752)
monthly_income = st.sidebar.number_input('Monthly Income', min_value=0, max_value=100000, value=25000)
country = st.sidebar.selectbox('Country', ['China', 'UK', 'India'], index=0)
state = st.sidebar.text_input('State', value='Guangdong')
city = st.sidebar.text_input('City', value='Dongguan')
category = st.sidebar.selectbox('Category', ['electronics', 'fashion', 'toys'], index=0)
product = st.sidebar.text_input('Product', value='table fan')
cost = st.sidebar.number_input('Cost', min_value=0, value=30)
price = st.sidebar.number_input('Price', min_value=0, value=50)
quantity = st.sidebar.number_input('Quantity', min_value=1, value=4)
campaign_schema = st.sidebar.selectbox('Campaign Schema', ['Instagram-ads', 'Google-ads', 'Facebook-ads', 'Twitter-ads', 'Billboard-QR code'], index=0)
payment_method = st.sidebar.selectbox('Payment Method', ['Cash On Delivery', 'Debit Card', 'Credit Card'], index=0)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'], index=1)

# Create a dictionary to hold user input data
input_data = {
    'Age': age,
    'CreditScore': credit_score,
    'MonthlyIncome': monthly_income,
    'Country': country,
    'State': state,
    'City': city,
    'Category': category,
    'Product': product,
    'Cost': cost,
    'Price': price,
    'Quantity': quantity,
    'CampaignSchema ': campaign_schema,
    'PaymentMethod': payment_method,
    'Gender': gender
}

# Predict the order confirmation based on user inputs
if st.sidebar.button('Predict'):
    prediction = predict_order_confirmation(input_data, model)
    confirmation = 'Confirmed' if prediction else 'Not Confirmed'
    st.sidebar.success(f'The predicted order confirmation for the product "{product}" is: {confirmation}')
    
    # Explanation of the result
    st.sidebar.markdown(f"### Explanation:")
    st.sidebar.markdown(f"- **Age**: {age}")
    st.sidebar.markdown(f"- **Credit Score**: {credit_score}")
    st.sidebar.markdown(f"- **Monthly Income**: {monthly_income}")
    st.sidebar.markdown(f"- **Country**: {country}")
    st.sidebar.markdown(f"- **State**: {state}")
    st.sidebar.markdown(f"- **City**: {city}")
    st.sidebar.markdown(f"- **Category**: {category}")
    st.sidebar.markdown(f"- **Product**: {product}")
    st.sidebar.markdown(f"- **Cost**: {cost}")
    st.sidebar.markdown(f"- **Price**: {price}")
    st.sidebar.markdown(f"- **Quantity**: {quantity}")
    st.sidebar.markdown(f"- **Campaign Schema**: {campaign_schema}")
    st.sidebar.markdown(f"- **Payment Method**: {payment_method}")
    st.sidebar.markdown(f"- **Gender**: {gender}")


# Load your dataset for visualization (assuming df is your loaded DataFrame)
df = pd.read_csv("https://raw.githubusercontent.com/kavyasri2099/insights_app/main/insights.csv")  # Adjust path as necessary

# Optionally, display all required visualizations
st.header('Data Visualizations')
# create_visualizations(df)

# Optionally, you can add more functionality like explanations, additional visualizations, etc.
