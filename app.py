import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\harti\src\student_data.csv')
df.columns = df.columns.str.strip()

df['result'] = df['result'].str.lower().str.strip()
df['result'] = df['result'].map({'pass': 1, 'fail': 0})

X = df[['study_hours', 'attendance', 'scores']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("\nEnter student details to predict result:")
study_hours = float(input("Study hours: "))
attendance = float(input("Attendance %: "))
scores = float(input("Previous scores: "))

new_data = pd.DataFrame([[study_hours, attendance, scores]], 
                        columns=['study_hours', 'attendance', 'scores'])

prediction = model.predict(new_data)[0]

if prediction == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")
