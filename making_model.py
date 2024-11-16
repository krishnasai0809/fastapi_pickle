import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Load the dataset
california_housing = fetch_california_housing()
X = california_housing.data  # Features (predictors)
y = california_housing.target  # Target variable (house prices)

# Select the first four features (predictors)
X = X[:, :4]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but often helps with regression models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('./dt_model_regression.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as 'dt_model_regression.pkl'.")
