import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model (trained on 4 features only)
model = joblib.load("my_model.pkl")

# Define feature input functions with range info
feature_inputs = {
    "Hours_Studied": lambda: st.number_input("Hours Studied (Weekly) [0-40]", min_value=0, max_value=40, value=10),
    "Attendance": lambda: st.number_input("Attendance (%) [0-100]", min_value=0, max_value=100, value=85),
    "Parental_Involvement": lambda: st.number_input("Parental Involvement [0=Low, 1=Med, 2=High]", min_value=0, max_value=2, value=1),
    "Access_to_Resources": lambda: st.number_input("Access to Learning Resources [0=None, 1=Some, 2=All]", min_value=0, max_value=2, value=2),
    "Extracurricular_Activities": lambda: st.number_input("Extracurricular Activities [0=No, 1=Yes]", min_value=0, max_value=1, value=1),
    "Sleep_Hours": lambda: st.number_input("Sleep Hours (Daily) [0-12]", min_value=0, max_value=12, value=7),
    "Previous_Scores": lambda: st.number_input("Previous Test Score [0-100]", min_value=0, max_value=100, value=75),
    "Motivation_Level": lambda: st.number_input("Motivation Level [0=Low, 1=Med, 2=High]", min_value=0, max_value=2, value=2),
    "Internet_Access": lambda: st.number_input("Internet Access [0=No, 1=Yes]", min_value=0, max_value=1, value=1),
    "Tutoring_Sessions": lambda: st.number_input("Monthly Tutoring Sessions [0-10]", min_value=0, max_value=10, value=2),
    "Family_Income": lambda: st.number_input("Family Income Level [0=Low, 1=Mid, 2=High]", min_value=0, max_value=2, value=1),
    "Teacher_Quality": lambda: st.number_input("Teacher Quality [0=Poor, 1=Avg, 2=Good]", min_value=0, max_value=2, value=2),
    "School_Type": lambda: st.number_input("School Type [0=Public, 1=Private]", min_value=0, max_value=1, value=1),
    "Peer_Influence": lambda: st.number_input("Peer Influence [0=Bad, 1=Neutral, 2=Good]", min_value=0, max_value=2, value=2),
    "Physical_Activity": lambda: st.number_input("Physical Activity (Hours/Week) [0-10]", min_value=0, max_value=10, value=2),
    "Learning_Disabilities": lambda: st.number_input("Learning Disabilities [0=No, 1=Yes]", min_value=0, max_value=1, value=0),
    "Parental_Education_Level": lambda: st.number_input("Parental Education Level [0=None, 1=School, 2=College]", min_value=0, max_value=2, value=2),
    "Distance_from_Home": lambda: st.number_input("Distance from Home [0=Near, 1=Moderate, 2=Far]", min_value=0, max_value=2, value=1),
    "Gender": lambda: st.number_input("Gender [0=Male, 1=Female]", min_value=0, max_value=1, value=0),
    "Exam_Score": lambda: st.number_input("Final Exam Score [0-100]", min_value=0, max_value=100, value=70)
}

st.title("🎓 Student Performance Predictor")

# Display class meaning
st.markdown("### 🧠 Prediction Classes:")
st.markdown("""
- **0 = Poor**
- **1 = Moderate**
- **2 = Excellent**
""")

# Feature selection
selected_features = st.multiselect(
    "📌 Select EXACTLY 4 features to use for prediction:",
    list(feature_inputs.keys())
)

if len(selected_features) < 4:
    st.warning("⚠️ Please select 4 features.")
elif len(selected_features) > 4:
    st.warning("⚠️ You've selected more than 4 features. Please select exactly 4.")
else:
    st.subheader("✍️ Enter values for the selected features:")
    input_values = [feature_inputs[feature]() for feature in selected_features]

    if st.button("🔮 Predict"):
        try:
            input_array = np.array([input_values])
            prediction = model.predict(input_array)[0]

            # Class meaning
            performance = {0: "Poor", 1: "Moderate", 2: "Excellent"}
            st.success(f"🎯 Predicted Result: **{performance[prediction]} (Class {prediction})**")

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_array)[0]
                st.info("📊 Prediction Probabilities:")
                for i, p in enumerate(prob):
                    st.write(f"Class {i} ({performance[i]}): {p:.2f}")

                # Plotting graph
                fig, ax = plt.subplots()
                ax.bar([f"{performance[i]} (Class {i})" for i in range(len(prob))], prob, color='skyblue')
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probability Distribution")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


