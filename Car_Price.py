import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

st.set_page_config(layout="wide")

with open('XGBoost.pkl', 'rb') as f:
    lm2 = pickle.load(f)

image_sidebar = Image.open('Pic 1.png') 
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Vehicle Features')

def get_user_input():
    horsepower = st.sidebar.number_input('Horsepower (No)', min_value=0, max_value=1000, step=1, value=300)
    torque = st.sidebar.number_input('Torque (No)', min_value=0, max_value=1500, step=1, value=400)
    
    make = st.sidebar.selectbox('Make', ['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])

    user_data = {
        'Horsepower_No': horsepower,
        'Torque_No': torque,
        f'Make_{make}': 1,
        f'Body Size_{body_size}': 1,
        f'Body Style_{body_style}': 1,
        f'Engine Aspiration_{engine_aspiration}': 1,
        f'Drivetrain_{drivetrain}': 1,
        f'Transmission_{transmission}': 1,
    }
    
    # Return both raw data and display data
    display_data = {
        'Make': make,
        'Body Size': body_size,
        'Body Style': body_style,
        'Engine Aspiration': engine_aspiration,
        'Drivetrain': drivetrain,
        'Transmission': transmission.capitalize(),
        'Horsepower': horsepower,
        'Torque': torque
    }
    
    return user_data, display_data

image_banner = Image.open('Pic 2.png') 
st.image(image_banner, use_container_width=True)

st.markdown("<h1 style='text-align: center;'>Vehicle Price Prediction App</h1>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

with left_col:
    st.header("📋 Feature Details")
    
    user_data, display_data = get_user_input()
    
    # Create a more presentable layout
    st.markdown("### Vehicle Specifications")
    
    # Performance specs
    st.markdown("**Performance**")
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.metric("Horsepower", f"{display_data['Horsepower']} HP")
    with perf_col2:
        st.metric("Torque", f"{display_data['Torque']} Nm")
    
    st.markdown("---")
    
    # Vehicle details
    st.markdown("**Vehicle Details**")
    details_df = pd.DataFrame({
        'Feature': ['Make', 'Body Size', 'Body Style', 'Drivetrain', 'Transmission', 'Engine Aspiration'],
        'Value': [
            display_data['Make'],
            display_data['Body Size'],
            display_data['Body Style'],
            display_data['Drivetrain'],
            display_data['Transmission'],
            display_data['Engine Aspiration']
        ]
    })
    
    st.dataframe(
        details_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium")
        }
    )

with right_col:
    st.header("💰 Predict Vehicle Price")
    
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    features = [
        'Horsepower_No', 'Torque_No', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
        'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
        'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van', 
        'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe', 
        'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
        'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
        'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
        'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
        'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged', 
        'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD', 
        'Transmission_automatic', 'Transmission_manual'
    ]

    st.markdown("Click the button below to get the estimated price for your selected vehicle configuration.")
    
    if st.button("🚀 Predict Price", use_container_width=True):
        input_array = prepare_input(user_data, features)
        prediction = lm2.predict(input_array)
        
        st.markdown("---")
        st.markdown("### Predicted Price")
        st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>${prediction[0]:,.2f}</h2>", unsafe_allow_html=True)
        st.success("Price prediction completed successfully!")