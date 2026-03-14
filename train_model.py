# train_model.py
# This script trains a machine learning model to predict fitness levels based on member data.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('dataset/fitness_members.csv')

# Map fitness levels to numeric values
fitness_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
data['Fitness_Level'] = data['Fitness_Level'].map(fitness_mapping)

# Define features and target
features = ['Age', 'Weight', 'Height', 'Workout_Hours', 'Calories_Intake', 'Heart_Rate', 'Steps_Per_Day']
X = data[features]
y = data['Fitness_Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'model.pkl')
print('Model saved as model.pkl')