import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('student_data.csv')
df.columns = df.columns.str.strip()

# Clean and encode target
df['result'] = df['result'].str.lower().str.strip()
df['result'] = df['result'].map({'pass': 1, 'fail': 0})

# Features and target
X = df[['study_hours', 'attendance', 'scores']]
y = df['result']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Streamlit UI ---
st.title("🎓 Student Performance Prediction")
st.write("Enter student details below to predict whether they will pass or fail.")

study_hours = st.number_input("Study hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
scores = st.number_input("Previous scores", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict Result"):
    new_data = pd.DataFrame([[study_hours, attendance, scores]],
                            columns=['study_hours', 'attendance', 'scores'])
    prediction = model.predict(new_data)[0]
    if prediction == 1:
        st.success("Student will PASS")
    else:
        st.error("Student will FAIL")
