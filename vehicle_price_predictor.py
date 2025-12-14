import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the vehicle dataset"""
    print("Loading vehicle dataset...")
    df = pd.read_csv('dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    print("\nCleaning data...")
    
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Convert year to age
    current_year = 2024
    data['age'] = current_year - data['year']
    
    # Extract make from name if make is missing
    data['make'] = data['make'].fillna(data['name'].str.split().str[1])
    
    print(f"Data cleaned. Shape: {data.shape}")
    return data

def exploratory_data_analysis(data):
    """Perform exploratory data analysis"""
    print("\nPerforming exploratory data analysis...")
    
    try:
        # Distribution of vehicle prices
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['price'], bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Vehicle Prices')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(data['price']), bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Log(Price)')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('price_distribution.png')
        plt.close()  # Close instead of show to avoid display issues
        
        # Top makes by count
        plt.figure(figsize=(12, 6))
        make_counts = data['make'].value_counts().head(15)
        sns.barplot(x=make_counts.values, y=make_counts.index)
        plt.title('Top 15 Vehicle Makes')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('top_makes.png')
        plt.close()
        
        # Price vs Age
        plt.figure(figsize=(10, 6))
        plt.scatter(data['age'], data['price'], alpha=0.5)
        plt.title('Vehicle Price vs Age')
        plt.xlabel('Age (years)')
        plt.ylabel('Price ($)')
        plt.tight_layout()
        plt.savefig('price_vs_age.png')
        plt.close()
        
        # Price by fuel type
        plt.figure(figsize=(10, 6))
        fuel_types = data['fuel'].unique()
        fuel_prices = [data[data['fuel'] == fuel]['price'].dropna() for fuel in fuel_types]
        plt.boxplot(fuel_prices, labels=fuel_types)
        plt.title('Vehicle Price by Fuel Type')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('price_by_fuel.png')
        plt.close()
        
        print("EDA plots saved successfully.")
    except Exception as e:
        print(f"Warning: Could not generate all plots: {e}")

def feature_engineering(data):
    """Create new features for better predictions"""
    print("\nPerforming feature engineering...")
    
    # Create a copy of the data
    df = data.copy()
    
    # Extract horsepower from engine description if available
    # This is a simplified approach - in practice, you'd want a more robust parser
    try:
        df['horsepower'] = df['engine'].str.extract(r'(\d+)\s*hp|\d+\s*horsepower', flags=re.IGNORECASE)[0]
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
    except Exception as e:
        print(f"Warning: Could not extract horsepower: {e}")
        df['horsepower'] = 150  # Default value
    
    # Create engine size feature
    try:
        df['engine_size'] = df['engine'].str.extract(r'(\d\.?\d?)\s*L', flags=re.IGNORECASE)[0]
        df['engine_size'] = pd.to_numeric(df['engine_size'], errors='coerce')
        df['engine_size'].fillna(df['engine_size'].median(), inplace=True)
    except Exception as e:
        print(f"Warning: Could not extract engine size: {e}")
        df['engine_size'] = 3.0  # Default value
    
    # Encode categorical variables
    try:
        le_make = LabelEncoder()
        df['make_encoded'] = le_make.fit_transform(df['make'])
        
        le_fuel = LabelEncoder()
        df['fuel_encoded'] = le_fuel.fit_transform(df['fuel'])
        
        le_transmission = LabelEncoder()
        df['transmission_encoded'] = le_transmission.fit_transform(df['transmission'])
        
        le_body = LabelEncoder()
        df['body_encoded'] = le_body.fit_transform(df['body'])
        
        le_drivetrain = LabelEncoder()
        df['drivetrain_encoded'] = le_drivetrain.fit_transform(df['drivetrain'])
    except Exception as e:
        print(f"Error encoding categorical variables: {e}")
        raise e
    
    print("Feature engineering completed.")
    return df, le_make, le_fuel, le_transmission, le_body, le_drivetrain

def prepare_features(df):
    """Prepare features for modeling"""
    # Select features for modeling
    feature_columns = [
        'year', 'age', 'mileage', 'doors', 'cylinders',
        'make_encoded', 'fuel_encoded', 'transmission_encoded', 
        'body_encoded', 'drivetrain_encoded', 'horsepower', 'engine_size'
    ]
    
    # Ensure all feature columns exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Remove any rows with missing values in feature columns
    df_clean = df.dropna(subset=available_features + ['price'])
    
    # If any features are missing, add them with default values
    for col in feature_columns:
        if col not in df_clean.columns:
            if col == 'horsepower':
                df_clean[col] = 150
            elif col == 'engine_size':
                df_clean[col] = 3.0
            else:
                df_clean[col] = 0
    
    X = df_clean[feature_columns]
    y = df_clean['price']
    
    print(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def train_models(X, y):
    """Train multiple regression models"""
    print("\nTraining models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        trained_models[name] = model
        
        print(f"{name} Results:")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  R²: {r2:.4f}")
    
    return trained_models, results, X_test, y_test, scaler, X_train, y_train

def visualize_results(results):
    """Visualize model performance"""
    print("\nVisualizing model performance...")
    
    try:
        # Create comparison plot
        metrics = ['RMSE', 'MAE', 'R2']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if metric == 'R2':
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{value:.3f}', ha='center', va='bottom')
                else:
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                                f'${value:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()  # Close instead of show to avoid display issues
        print("Model comparison plot saved successfully.")
    except Exception as e:
        print(f"Warning: Could not generate model comparison plot: {e}")

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for the Random Forest model"""
    print("\nPerforming hyperparameter tuning for Random Forest...")
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize Random Forest
    rf = RandomForestRegressor(random_state=42)
    
    # Since we're not using GridSearchCV to keep dependencies minimal,
    # we'll manually test a few combinations
    best_score = -float('inf')
    best_params = None
    best_model = None
    
    # Test a subset of parameter combinations
    test_combinations = [
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 5}
    ]
    
    for params in test_combinations:
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    
    print(f"Best parameters: {best_params}")
    print(f"Best training score: {best_score:.4f}")
    
    return best_model, best_params

def predict_price(model, scaler, encoders, feature_names, year, make, mileage, fuel, transmission, body, drivetrain, cylinders=4, doors=4, horsepower=150, engine_size=3.0):
    """Predict the price of a vehicle based on its features"""
    # Encode categorical variables
    try:
        make_encoded = encoders['make'].transform([make])[0]
    except ValueError:
        print(f"Warning: Make '{make}' not seen during training. Using default value.")
        make_encoded = 0
    
    try:
        fuel_encoded = encoders['fuel'].transform([fuel])[0]
    except ValueError:
        print(f"Warning: Fuel type '{fuel}' not seen during training. Using default value.")
        fuel_encoded = 0
    
    try:
        transmission_encoded = encoders['transmission'].transform([transmission])[0]
    except ValueError:
        print(f"Warning: Transmission '{transmission}' not seen during training. Using default value.")
        transmission_encoded = 0
    
    try:
        body_encoded = encoders['body'].transform([body])[0]
    except ValueError:
        print(f"Warning: Body type '{body}' not seen during training. Using default value.")
        body_encoded = 0
    
    try:
        drivetrain_encoded = encoders['drivetrain'].transform([drivetrain])[0]
    except ValueError:
        print(f"Warning: Drivetrain '{drivetrain}' not seen during training. Using default value.")
        drivetrain_encoded = 0
    
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

def interactive_prediction(model, scaler, encoders, feature_names):
    """Interactive interface for predicting vehicle prices"""
    print("\n=== Vehicle Price Predictor ===")
    print("Enter vehicle details to predict its price:")
    
    try:
        year = int(input("Year: "))
        make = input("Make (e.g., Ford, Toyota, BMW): ")
        mileage = float(input("Mileage: "))
        fuel = input("Fuel type (e.g., Gasoline, Diesel, Electric): ")
        transmission = input("Transmission (e.g., Automatic, Manual): ")
        body = input("Body style (e.g., SUV, Sedan, Pickup Truck): ")
        drivetrain = input("Drivetrain (e.g., All-wheel Drive, Front-wheel Drive): ")
        
        # Optional inputs with defaults
        cylinders_input = input("Number of cylinders (default 4): ")
        cylinders = int(cylinders_input) if cylinders_input else 4
        
        doors_input = input("Number of doors (default 4): ")
        doors = int(doors_input) if doors_input else 4
        
        horsepower_input = input("Horsepower (default 150): ")
        horsepower = float(horsepower_input) if horsepower_input else 150
        
        engine_size_input = input("Engine size in liters (default 3.0): ")
        engine_size = float(engine_size_input) if engine_size_input else 3.0
        
        # Predict price
        predicted_price = predict_price(model, scaler, encoders, feature_names, year, make, mileage, 
                                     fuel, transmission, body, drivetrain, cylinders, doors, 
                                     horsepower, engine_size)
        
        print(f"\nPredicted Price: ${predicted_price:,.2f}")
        
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")
    except Exception as e:
        print(f"Error during prediction: {e}")

def save_model_and_encoders(model, encoders, scaler, feature_names):
    """Save the trained model and encoders for later use"""
    print("\nSaving model and encoders...")
    
    # Save the model
    with open('vehicle_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model and encoders saved successfully.")

def visualize_feature_importance(model, feature_names):
    """Visualize feature importance from the trained model"""
    print("\nCreating feature importance visualization...")
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances in Vehicle Price Prediction")
        bars = plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved successfully.")
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot: {e}")

def main():
    """Main function to run the vehicle price prediction pipeline"""
    print("=== Vehicle Price Prediction System ===")
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Clean data
    cleaned_data = clean_data(df)
    
    # Perform exploratory data analysis
    exploratory_data_analysis(cleaned_data)
    
    # Feature engineering
    try:
        engineered_data, le_make, le_fuel, le_transmission, le_body, le_drivetrain = feature_engineering(cleaned_data)
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        return
    
    # Store encoders for later use in prediction
    encoders = {
        'make': le_make,
        'fuel': le_fuel,
        'transmission': le_transmission,
        'body': le_body,
        'drivetrain': le_drivetrain
    }
    
    # Prepare features
    X, y = prepare_features(engineered_data)
    
    # Train models
    models, results, X_test, y_test, scaler, X_train, y_train = train_models(X, y)
    
    # Visualize results
    visualize_results(results)
    
    # Print best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['R2']:.4f}")
    print(f"RMSE: ${results[best_model_name]['RMSE']:,.2f}")
    
    # Hyperparameter tuning for the best model
    if best_model_name == 'Random Forest':
        tuned_model, best_params = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate tuned model
        y_pred_tuned = tuned_model.predict(X_test)
        rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
        r2_tuned = r2_score(y_test, y_pred_tuned)
        
        print(f"\nTuned Random Forest Results:")
        print(f"  RMSE: ${rmse_tuned:,.2f}")
        print(f"  R²: {r2_tuned:.4f}")
        
        # Update best model if tuned model is better
        if r2_tuned > results[best_model_name]['R2']:
            best_model = tuned_model
            print("Tuned model performs better. Using tuned model.")
        else:
            print("Original model performs better or equally. Using original model.")
    
    # Visualize feature importance
    feature_names = [
        'year', 'age', 'mileage', 'doors', 'cylinders',
        'make_encoded', 'fuel_encoded', 'transmission_encoded', 
        'body_encoded', 'drivetrain_encoded', 'horsepower', 'engine_size'
    ]
    visualize_feature_importance(best_model, feature_names)
    
    # Save model and encoders for later use
    save_model_and_encoders(best_model, encoders, scaler, feature_names)
    
    # Interactive prediction interface
    while True:
        choice = input("\nWould you like to predict a vehicle price? (y/n): ").lower().strip()
        if choice == 'y' or choice == 'yes':
            interactive_prediction(best_model, scaler, encoders, feature_names)
        else:
            break
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()