# fitness_dashboard.py
# This is a Streamlit web app for the Fitness Center Health Monitoring System.

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('model.pkl')

# Load the dataset for visualizations
data = pd.read_csv('dataset/fitness_members.csv')

# Reverse mapping for fitness levels
fitness_mapping_reverse = {0: 'Beginner', 1: 'Intermediate', 2: 'Advanced'}

# Title
st.title("Fitness Center Health Monitoring System")

# Input fields
st.header("Enter Member Details")
age = st.number_input("Age", min_value=18, max_value=100, value=25)
weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0)
height = st.number_input("Height (cm)", min_value=140.0, max_value=220.0, value=170.0)
workout_hours = st.number_input("Workout Hours per week", min_value=0.0, max_value=20.0, value=5.0)
calories_intake = st.number_input("Calories Intake per day", min_value=1000, max_value=4000, value=2000)
heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=50, max_value=120, value=70)
steps_per_day = st.number_input("Steps Per Day", min_value=0, max_value=30000, value=8000)

# Predict button
if st.button("Predict Fitness Level"):
    # Prepare input data
    input_data = np.array([age, weight, height, workout_hours, calories_intake, heart_rate, steps_per_day]).reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data)[0]
    predicted_level = fitness_mapping_reverse[prediction]
    st.success(f"Predicted Fitness Level: {predicted_level}")

# Visualizations section
st.header("Data Visualizations")

# Histogram of member ages
st.subheader("Histogram of Member Ages")
fig, ax = plt.subplots()
ax.hist(data['Age'], bins=20, edgecolor='black')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Member Ages')
st.pyplot(fig)

# Scatter plot of steps vs calories
st.subheader("Scatter Plot of Steps per Day vs Calories Intake")
fig, ax = plt.subplots()
ax.scatter(data['Steps_Per_Day'], data['Calories_Intake'], alpha=0.5)
ax.set_xlabel('Steps Per Day')
ax.set_ylabel('Calories Intake')
ax.set_title('Steps vs Calories Scatter Plot')
st.pyplot(fig)

# Distribution of fitness levels
st.subheader("Distribution of Fitness Levels")
fig, ax = plt.subplots()
sns.countplot(x='Fitness_Level', data=data, ax=ax)
ax.set_xlabel('Fitness Level')
ax.set_ylabel('Count')
ax.set_title('Distribution of Fitness Levels')
st.pyplot(fig)