# predict_fitness.py
# This is a console-based program to predict a member's fitness level using the trained model.

import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Reverse mapping for fitness levels
fitness_mapping_reverse = {0: 'Beginner', 1: 'Intermediate', 2: 'Advanced'}

# Function to get user input
def get_user_input():
    print("Enter member details:")
    member_id = input("Member ID: ")
    age = float(input("Age: "))
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    workout_hours = float(input("Workout Hours per week: "))
    calories_intake = float(input("Calories Intake per day: "))
    heart_rate = float(input("Resting Heart Rate (bpm): "))
    steps_per_day = float(input("Steps Per Day: "))
    return [age, weight, height, workout_hours, calories_intake, heart_rate, steps_per_day]

# Get user input
user_data = get_user_input()

# Convert to numpy array and reshape
user_data = np.array(user_data).reshape(1, -1)

# Make prediction
prediction = model.predict(user_data)[0]

# Display the result
predicted_level = fitness_mapping_reverse[prediction]
print(f"Predicted Fitness Level: {predicted_level}")