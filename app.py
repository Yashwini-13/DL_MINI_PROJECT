import numpy as np
import joblib
import gradio as gr
from tensorflow import keras

model = keras.models.load_model("student_performance_model.keras")
scaler = joblib.load("scaler.pkl")

def predict_ui(attendance, assignment, internal, study_hours, gpa):
    data = scaler.transform(np.array([[attendance, assignment, internal, study_hours, gpa]]))
    prob = float(model.predict(data, verbose=0)[0][0])
    if prob >= 0.70:
        level, advice = "HIGH", "Immediate counselling and remedial classes"
    elif prob >= 0.40:
        level, advice = "MEDIUM", "Mentor follow-up and personalised study plan"
    else:
        level, advice = "LOW", "Regular monitoring"
    return f"{prob*100:.2f}%", level, advice

demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Slider(0, 100, value=75, label="Attendance (%)"),
        gr.Slider(0, 100, value=65, label="Assignment Marks"),
        gr.Slider(0, 100, value=60, label="Internal Score"),
        gr.Slider(0, 40, value=12, label="Study Hours per Week"),
        gr.Slider(0, 10, value=6.5, step=0.1, label="Previous GPA"),
    ],
    outputs=[
        gr.Textbox(label="Risk Probability"),
        gr.Textbox(label="Risk Level"),
        gr.Textbox(label="Recommendation"),
    ],
    title="Student Academic Performance Predictor",
    description="Enter a student's details to predict the risk of failing.",
)

if __name__ == "__main__":
    demo.launch(share=True)