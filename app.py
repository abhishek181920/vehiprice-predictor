import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open('vehicle_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, encoders, scaler, feature_names

# Function to predict vehicle price
def predict_price(model, encoders, scaler, feature_names, year, make, mileage, fuel, transmission, body, drivetrain, cylinders=4, doors=4, horsepower=150, engine_size=3.0):
    """Predict the price of a vehicle based on its features"""
    # Encode categorical variables
    try:
        make_encoded = encoders['make'].transform([make])[0]
    except ValueError:
        make_encoded = 0  # Default value for unseen categories
    
    try:
        fuel_encoded = encoders['fuel'].transform([fuel])[0]
    except ValueError:
        fuel_encoded = 0  # Default value for unseen categories
    
    try:
        transmission_encoded = encoders['transmission'].transform([transmission])[0]
    except ValueError:
        transmission_encoded = 0  # Default value for unseen categories
    
    try:
        body_encoded = encoders['body'].transform([body])[0]
    except ValueError:
        body_encoded = 0  # Default value for unseen categories
    
    try:
        drivetrain_encoded = encoders['drivetrain'].transform([drivetrain])[0]
    except ValueError:
        drivetrain_encoded = 0  # Default value for unseen categories
    
    # Calculate age
    current_year = 2024
    age = current_year - year
    
    # Create feature array
    features = np.array([[year, age, mileage, doors, cylinders,
                         make_encoded, fuel_encoded, transmission_encoded,
                         body_encoded, drivetrain_encoded, horsepower, engine_size]])
    
    # Create DataFrame with proper column names
    feature_df = pd.DataFrame(features, columns=feature_names)
    
    # Predict price
    predicted_price = model.predict(feature_df)[0]
    
    return predicted_price

# Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="wide")
    
    st.title("🚗 Vehicle Price Predictor")
    st.markdown("""
    This app predicts the price of a vehicle based on its specifications.
    Enter the vehicle details below to get a price estimate.
    """)
    
    # Load model and encoders
    try:
        model, encoders, scaler, feature_names = load_model_and_encoders()
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Create input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2020, step=1)
        make = st.text_input("Make", placeholder="e.g., Ford, Toyota, BMW")
        mileage = st.number_input("Mileage", min_value=0, value=30000, step=1000)
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
    
    with col2:
        st.subheader("Specifications")
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Other"])
        body = st.selectbox("Body Style", ["SUV", "Sedan", "Pickup Truck", "Hatchback", "Coupe", "Van", "Wagon", "Convertible"])
        drivetrain = st.selectbox("Drivetrain", ["All-wheel Drive", "Front-wheel Drive", "Rear-wheel Drive", "Four-wheel Drive"])
        cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=12, value=4, step=1)
    
    with col3:
        st.subheader("Additional Details")
        doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4, step=1)
        horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=150, step=10)
        engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
        
        # Prediction button
        st.write("")
        st.write("")
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            if not make:
                st.warning("Please enter the vehicle make.")
                return
            
            # Make prediction
            try:
                predicted_price = predict_price(
                    model, encoders, scaler, feature_names,
                    year, make, mileage, fuel, transmission, body, drivetrain,
                    cylinders, doors, horsepower, engine_size
                )
                
                # Display result
                st.success(f"### Predicted Price: ${predicted_price:,.2f}")
                
                # Show additional information
                st.info(f"""
                **Vehicle Details:**
                - Year: {year}
                - Make: {make}
                - Mileage: {mileage:,} miles
                - Fuel Type: {fuel}
                - Transmission: {transmission}
                - Body Style: {body}
                - Drivetrain: {drivetrain}
                - Cylinders: {cylinders}
                - Doors: {doors}
                - Horsepower: {horsepower} HP
                - Engine Size: {engine_size} L
                """)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Add information about the model
    st.markdown("---")
    st.subheader("ℹ️ About This Model")
    st.markdown("""
    This vehicle price prediction model was trained on a dataset of vehicle listings.
    The model uses features such as year, make, mileage, fuel type, transmission, 
    body style, drivetrain, and engine specifications to estimate vehicle prices.
    
    **Model Performance:**
    - Algorithm: Random Forest
    - R² Score: ~0.48
    - RMSE: ~$11,466
    
    *Note: Predictions are estimates and may not reflect actual market prices.*
    """)

if __name__ == "__main__":
    main()