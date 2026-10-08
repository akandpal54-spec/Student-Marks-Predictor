# ============================================================
# STUDENT MARKS PREDICTOR
# B.Tech AI/ML Mini Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tkinter as tk
from tkinter import messagebox


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("STUDENT MARKS PREDICTOR")
print("=" * 60)

print("\nLoading dataset...")

data = pd.read_csv("student_data.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("\nDataset information:")
print(data.info())

print("\nChecking missing values:")
print(data.isnull().sum())


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignments"
    ]
]

y = data["final_marks"]


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. CREATE MACHINE LEARNING MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining machine learning model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 7. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

print("=" * 60)


# ============================================================
# 9. DISPLAY MODEL COEFFICIENTS
# ============================================================

print("\nFeature importance coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):

    print(
        f"{feature}: {coefficient:.4f}"
    )

print(f"\nIntercept: {model.intercept_:.4f}")


# ============================================================
# 10. PREDICTION FUNCTION
# ============================================================

def predict_marks():

    try:

        # Get values from input boxes

        study_hours = float(
            study_entry.get()
        )

        attendance = float(
            attendance_entry.get()
        )

        previous_marks = float(
            previous_entry.get()
        )

        assignments = float(
            assignment_entry.get()
        )


        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        if study_hours < 0 or study_hours > 24:

            messagebox.showerror(
                "Invalid Input",
                "Study hours must be between 0 and 24."
            )

            return


        if attendance < 0 or attendance > 100:

            messagebox.showerror(
                "Invalid Input",
                "Attendance must be between 0 and 100."
            )

            return


        if previous_marks < 0 or previous_marks > 100:

            messagebox.showerror(
                "Invalid Input",
                "Previous marks must be between 0 and 100."
            )

            return


        if assignments < 0 or assignments > 10:

            messagebox.showerror(
                "Invalid Input",
                "Assignments must be between 0 and 10."
            )

            return


        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        student = pd.DataFrame({

            "study_hours": [study_hours],

            "attendance": [attendance],

            "previous_marks": [previous_marks],

            "assignments": [assignments]

        })


        # ----------------------------------------------------
        # Predict marks
        # ----------------------------------------------------

        prediction = model.predict(student)[0]


        # Keep prediction between 0 and 100

        prediction = max(
            0,
            min(100, prediction)
        )


        # ----------------------------------------------------
        # Determine performance
        # ----------------------------------------------------

        if prediction >= 90:

            performance = "Excellent"

        elif prediction >= 75:

            performance = "Very Good"

        elif prediction >= 60:

            performance = "Good"

        elif prediction >= 40:

            performance = "Average"

        else:

            performance = "Needs Improvement"


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        result_label.config(

            text=
            f"Predicted Marks: {prediction:.2f}/100\n\n"
            f"Performance: {performance}",

            fg="green"

        )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter numbers only."
        )


# ============================================================
# 11. CREATE GRAPH
# ============================================================

def show_graph():

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test,
        y_pred
    )

    plt.xlabel("Actual Marks")

    plt.ylabel("Predicted Marks")

    plt.title(
        "Actual vs Predicted Marks"
    )

    plt.grid(True)

    plt.show()


# ============================================================
# 12. CREATE GUI
# ============================================================

root = tk.Tk()

root.title(
    "Student Marks Predictor"
)

root.geometry(
    "550x700"
)

root.resizable(
    False,
    False
)


# ============================================================
# 13. TITLE
# ============================================================

title_label = tk.Label(

    root,

    text="Student Marks Predictor",

    font=("Arial", 24, "bold")

)

title_label.pack(
    pady=20
)


# ============================================================
# 14. DESCRIPTION
# ============================================================

description = tk.Label(

    root,

    text=
    "Machine Learning Project\n"
    "Linear Regression",

    font=("Arial", 11)

)

description.pack(
    pady=5
)


# ============================================================
# 15. STUDY HOURS
# ============================================================

study_label = tk.Label(

    root,

    text="Daily Study Hours",

    font=("Arial", 12)

)

study_label.pack(
    pady=(20, 5)
)


study_entry = tk.Entry(

    root,

    width=25,

    font=("Arial", 13)

)

study_entry.pack()


# ============================================================
# 16. ATTENDANCE
# ============================================================

attendance_label = tk.Label(

    root,

    text="Attendance Percentage",

    font=("Arial", 12)

)

attendance_label.pack(
    pady=(15, 5)
)


attendance_entry = tk.Entry(

    root,

    width=25,

    font=("Arial", 13)

)

attendance_entry.pack()


# ============================================================
# 17. PREVIOUS MARKS
# ============================================================

previous_label = tk.Label(

    root,

    text="Previous Exam Marks",

    font=("Arial", 12)

)

previous_label.pack(
    pady=(15, 5)
)


previous_entry = tk.Entry(

    root,

    width=25,

    font=("Arial", 13)

)

previous_entry.pack()


# ============================================================
# 18. ASSIGNMENTS
# ============================================================

assignment_label = tk.Label(

    root,

    text="Assignments Completed (0-10)",

    font=("Arial", 12)

)

assignment_label.pack(
    pady=(15, 5)
)


assignment_entry = tk.Entry(

    root,

    width=25,

    font=("Arial", 13)

)

assignment_entry.pack()


# ============================================================
# 19. PREDICT BUTTON
# ============================================================

predict_button = tk.Button(

    root,

    text="Predict Marks",

    command=predict_marks,

    font=("Arial", 14, "bold"),

    padx=25,

    pady=10

)

predict_button.pack(
    pady=25
)


# ============================================================
# 20. GRAPH BUTTON
# ============================================================

graph_button = tk.Button(

    root,

    text="Show Model Graph",

    command=show_graph,

    font=("Arial", 11),

    padx=15,

    pady=7

)

graph_button.pack(
    pady=5
)


# ============================================================
# 21. RESULT
# ============================================================

result_label = tk.Label(

    root,

    text="Enter student details\n"
         "and click Predict Marks",

    font=("Arial", 15, "bold"),

    justify="center"

)

result_label.pack(
    pady=20
)


# ============================================================
# 22. START APPLICATION
# ============================================================

root.mainloop()