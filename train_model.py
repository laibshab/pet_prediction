import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv("pet_dataset.csv")

# Encode categorical variables
data['Breed_Size'] = data['Breed_Size'].map({'small': 1, 'medium': 2, 'large': 3})
data['Activity_Level'] = data['Activity_Level'].map({'low': 1, 'medium': 2, 'high': 3})

# Define features and target variable
X = data[['Age', 'Weight', 'Breed_Size', 'Activity_Level']]
y = data['Food_Consumption']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save trained model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)