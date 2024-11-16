import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Function to load the pickled model
def load_model():
    with open('dt_model_regression.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions using the loaded model
def predict(data):
    model = load_model()
    # Assuming 'data' is a pandas DataFrame for prediction
    prediction = model.predict(data)
    return prediction.tolist()  # Convert to list for easier return in FastAPI response
