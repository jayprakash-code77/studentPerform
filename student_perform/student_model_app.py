# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Set page config
# st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# # Load model function with caching
# @st.cache_resource
# def load_model():
#     return joblib.load('my_model.pkl')

# model = load_model()

# # Title and description
# st.title("🎓 Student Performance Prediction")
# st.write("""
# This app predicts student performance based on input parameters.
# Adjust the values using the sliders and dropdowns below.
# """)

# # Input section in columns
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Academic Factors")
#     hours_studied = st.slider("Weekly study hours", 0, 40, 10)
#     prev_scores = st.slider("Previous test scores", 0, 100, 75)
#     attendance = st.slider("Attendance percentage", 0, 100, 85)

# with col2:
#     st.subheader("Personal Factors")
#     sleep_hours = st.slider("Daily sleep hours", 0, 12, 7)
#     extracurricular = st.selectbox("Extracurricular activities", ["None", "1-2", "3+"])
#     meals_per_day = st.slider("Meals per day", 1, 5, 3)

# with col3:
#     st.subheader("Environmental Factors")
#     travel_time = st.selectbox("Commute time (mins)", ["<15", "15-30", "30-60", ">60"])
#     internet_access = st.radio("Internet access", ["Yes", "No"])
#     family_support = st.radio("Family support", ["Yes", "No"])

# # Preprocess inputs
# def preprocess_inputs():
#     # Convert categorical to numerical
#     extracurricular_map = {"None": 0, "1-2": 1, "3+": 2}
#     travel_time_map = {"<15": 0, "15-30": 1, "30-60": 2, ">60": 3}
    
#     input_dict = {
#         'hours_studied': hours_studied,
#         'previous_scores': prev_scores,
#         'attendance': attendance,
#         'sleep_hours': sleep_hours,
#         'extracurricular': extracurricular_map[extracurricular],
#         'meals_per_day': meals_per_day,
#         'travel_time': travel_time_map[travel_time],
#         'internet_access': 1 if internet_access == "Yes" else 0,
#         'family_support': 1 if family_support == "Yes" else 0
#     }
    
#     return pd.DataFrame([input_dict])

# # Prediction and results
# if st.button("Predict Performance", type="primary"):
#     input_df = preprocess_inputs()
    
#     with st.spinner('Making prediction...'):
#         prediction = model.predict(input_df)
#         proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
    
#     st.success("Prediction complete!")
    
#     # Display results
#     st.subheader("📊 Prediction Results")
    
#     if isinstance(prediction[0], (np.floating, float)):  # Regression case
#         st.metric("Predicted Score", f"{prediction[0]:.1f}/100")
        
#         # Visualize with gauge chart
#         fig, ax = plt.subplots(figsize=(3, 3))
#         ax.axis('equal')
#         ax.pie([prediction[0], 100-prediction[0]], 
#                labels=['Achieved', 'Remaining'],
#                colors=['#4CAF50', '#E0E0E0'],
#                startangle=90)
#         ax.add_artist(plt.Circle((0,0), 0.7, color='white'))
#         st.pyplot(fig)
        
#     else:  # Classification case
#         st.metric("Predicted Performance", prediction[0])
        
#         if proba is not None:
#             st.write("Prediction probabilities:")
#             proba_df = pd.DataFrame({
#                 'Class': model.classes_,
#                 'Probability': proba[0]
#             }).sort_values('Probability', ascending=False)
#             st.bar_chart(proba_df.set_index('Class'))
    
#     # Show interpretation
#     with st.expander("ℹ️ Interpretation"):
#         st.write("""
#         - Scores above 80 indicate excellent performance
#         - Scores between 60-80 indicate average performance
#         - Scores below 60 may need intervention
#         """)

# # Optional: Add sample data
# with st.expander("💡 Sample Input Values"):
#     st.write("""
#     | Parameter          | Good Performance Range |
#     |--------------------|------------------------|
#     | Study Hours        | 15-25 hrs/week        |
#     | Previous Scores    | 70+                   |
#     | Sleep Hours        | 7-9 hrs/day           |
#     """)

# # Run with: streamlit run student_model_app.py















































# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Set page config
# st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# # Load model function with caching
# @st.cache_resource
# def load_model():
#     return joblib.load('my_model.pkl')

# model = load_model()

# # Title and description
# st.title("🎓 Student Performance Prediction")
# st.write("""
# This app predicts student performance based on input parameters.
# Adjust the values using the sliders and dropdowns below.
# """)

# # Input section - ONLY THE 4 FEATURES YOUR MODEL EXPECTS
# st.subheader("Input Parameters")
# col1, col2 = st.columns(2)

# with col1:
#     # Feature 1
#     hours_studied = st.slider("Weekly study hours", 0, 40, 10)
#     # Feature 2
#     prev_scores = st.slider("Previous test scores", 0, 100, 75)

# with col2:
#     # Feature 3
#     sleep_hours = st.slider("Daily sleep hours", 0, 12, 7)
#     # Feature 4
#     extracurricular = st.selectbox("Extracurricular activities", ["None", "1-2", "3+"])

# # Preprocess inputs (ONLY the 4 features)
# def preprocess_inputs():
#     extracurricular_map = {"None": 0, "1-2": 1, "3+": 2}
    
#     input_dict = {
#         'hours_studied': hours_studied,
#         'previous_scores': prev_scores,
#         'sleep_hours': sleep_hours,
#         'extracurricular': extracurricular_map[extracurricular]
#     }
    
#     return pd.DataFrame([input_dict])

# # Prediction and results
# if st.button("Predict Performance", type="primary"):
#     input_df = preprocess_inputs()
    
#     # Debug: Show what's being sent to the model
#     st.write("Features being sent to model:", input_df.columns.tolist())
    
#     with st.spinner('Making prediction...'):
#         prediction = model.predict(input_df)
#         proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
    
#     st.success("Prediction complete!")
    
#     # Display results
#     st.subheader("📊 Prediction Results")
    
#     if isinstance(prediction[0], (np.floating, float)):  # Regression case
#         st.metric("Predicted Score", f"{prediction[0]:.1f}/100")
        
#         # Visualize with gauge chart
#         fig, ax = plt.subplots(figsize=(3, 3))
#         ax.axis('equal')
#         ax.pie([prediction[0], 100-prediction[0]], 
#                labels=['Achieved', 'Remaining'],
#                colors=['#4CAF50', '#E0E0E0'],
#                startangle=90)
#         ax.add_artist(plt.Circle((0,0), 0.7, color='white'))
#         st.pyplot(fig)
        
#     else:  # Classification case
#         st.metric("Predicted Performance", prediction[0])
        
#         if proba is not None:
#             st.write("Prediction probabilities:")
#             proba_df = pd.DataFrame({
#                 'Class': model.classes_,
#                 'Probability': proba[0]
#             }).sort_values('Probability', ascending=False)
#             st.bar_chart(proba_df.set_index('Class'))
    
#     # Show interpretation
#     with st.expander("ℹ️ Interpretation"):
#         st.write("""
#         - Scores above 80 indicate excellent performance
#         - Scores between 60-80 indicate average performance
#         - Scores below 60 may need intervention
#         """)

# # Debugging help
# with st.expander("🔍 Model Information"):
#     try:
#         st.write(f"Model expects {model.n_features_in_} features")
#         st.write("Expected feature names (if available):", getattr(model, 'feature_names_in_', "Not available"))
#     except Exception as e:
#         st.write("Could not retrieve model info:", str(e))
        




























# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load the trained model
# @st.cache_resource
# def load_model():
#     return joblib.load('my_model.pkl')  # Update with your actual filename

# model = load_model()

# # App title and description
# st.title("📚 Student Performance Predictor")
# st.write("""
# Predict student performance based on key academic factors.
# Adjust the inputs below and click 'Predict'.
# """)

# # Input section - MODIFY THESE TO MATCH YOUR 4 FEATURES
# st.header("Student Data Input")
# col1, col2 = st.columns(2)

# with col1:
#     # Feature 1 (Example - replace with your actual features)
#     study_hours = st.slider("Weekly Study Hours", 0, 40, 10)
#     # Feature 2
#     prev_scores = st.slider("Previous Test Scores", 0, 100, 75)
    
# with col2:
#     # Feature 3
#     attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
#     # Feature 4
#     sleep_hours = st.slider("Daily Sleep Hours", 0, 12, 7)

# # Prediction function
# def make_prediction():
#     # CRITICAL: Array must match TRAINING DATA ORDER
#     input_data = np.array([[
#         study_hours,    # Must be first if this was first in training
#         prev_scores,    # Second feature
#         attendance,     # Third feature
#         sleep_hours     # Fourth feature
#     ]])
    
#     prediction = model.predict(input_data)
#     probabilities = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
    
#     return prediction[0], probabilities[0] if probabilities is not None else None

# # Prediction button
# if st.button("Predict Performance"):
#     try:
#         pred, probs = make_prediction()
        
#         # Display results
#         st.success(f"Predicted Performance: **{pred}**")
        
#         # Show probabilities if available
#         if probs is not None:
#             st.subheader("Class Probabilities")
#             prob_df = pd.DataFrame({
#                 'Performance Level': model.classes_,
#                 'Probability': probs
#             })
#             st.bar_chart(prob_df.set_index('Performance Level'))
            
#     except Exception as e:
#         st.error(f"Prediction failed: {str(e)}")
#         st.write("Common fixes:")
#         st.write("1. Ensure feature order matches training")
#         st.write("2. Verify all features are numerical")
#         st.write("3. Check model expects 4 features")

# # Debug panel
# with st.expander("🔍 Model Information"):
#     try:
#         st.write(f"Model type: {type(model).__name__}")
#         if hasattr(model, 'feature_names_in_'):
#             st.write("Expected features:", list(model.feature_names_in_))
#         st.write(f"Model classes: {model.classes_}")
#     except:
#         st.warning("Couldn't retrieve full model info")

# # How to use
# with st.expander("💡 Usage Guide"):
#     st.write("""
#     1. Adjust all sliders to match student data
#     2. Click 'Predict Performance'
#     3. View predicted class and probabilities
#     """)





























# import streamlit as st
# import numpy as np
# import joblib

# # Load trained model (trained on 4 features only)
# model = joblib.load("my_model.pkl")

# # Define feature input functions (lazy evaluation)
# feature_inputs = {
#     "Hours_Studied": lambda: st.slider("Hours Studied (Weekly)", 0, 40, 10),
#     "Attendance": lambda: st.slider("Attendance (%)", 0, 100, 85),
#     "Parental_Involvement": lambda: st.selectbox("Parental Involvement", [0, 1, 2]),
#     "Access_to_Resources": lambda: st.selectbox("Access to Learning Resources", [0, 1, 2]),
#     "Extracurricular_Activities": lambda: st.selectbox("Extracurricular Activities", [0, 1]),
#     "Sleep_Hours": lambda: st.slider("Sleep Hours (Daily)", 0, 12, 7),
#     "Previous_Scores": lambda: st.slider("Previous Test Score", 0, 100, 75),
#     "Motivation_Level": lambda: st.selectbox("Motivation Level", [0, 1, 2]),
#     "Internet_Access": lambda: st.selectbox("Internet Access", [0, 1]),
#     "Tutoring_Sessions": lambda: st.slider("Monthly Tutoring Sessions", 0, 10, 2),
#     "Family_Income": lambda: st.selectbox("Family Income Level", [0, 1, 2]),
#     "Teacher_Quality": lambda: st.selectbox("Teacher Quality", [0, 1, 2]),
#     "School_Type": lambda: st.selectbox("School Type", [0, 1]),
#     "Peer_Influence": lambda: st.selectbox("Peer Influence", [0, 1, 2]),
#     "Physical_Activity": lambda: st.slider("Physical Activity (Hours/Week)", 0, 10, 2),
#     "Learning_Disabilities": lambda: st.selectbox("Learning Disabilities", [0, 1]),
#     "Parental_Education_Level": lambda: st.selectbox("Parental Education Level", [0, 1, 2]),
#     "Distance_from_Home": lambda: st.selectbox("Distance from Home (Level)", [0, 1, 2]),
#     "Gender": lambda: st.selectbox("Gender", [0, 1]),
#     "Exam_Score": lambda: st.slider("Final Exam Score", 0, 100, 70)
# }

# st.title("🎓 Student Performance Predictor")

# # Select exactly 4 features
# selected_features = st.multiselect(
#     "📌 Select EXACTLY 4 features to use for prediction:",
#     list(feature_inputs.keys())
# )

# if len(selected_features) < 4:
#     st.warning("Please select 4 features.")
# elif len(selected_features) > 4:
#     st.warning("You've selected more than 4 features. Please select exactly 4.")
# else:
#     st.subheader("✍️ Enter values for the selected features:")
#     input_values = [feature_inputs[feature]() for feature in selected_features]
    
#     if st.button("🔮 Predict"):
#         try:
#             input_array = np.array([input_values])  # Shape: (1, 4)
#             prediction = model.predict(input_array)[0]
#             st.success(f"🎯 Predicted Result: **{prediction}**")

#             if hasattr(model, "predict_proba"):
#                 prob = model.predict_proba(input_array)[0]
#                 st.info(f"📊 Prediction Probabilities: {prob}")
#         except Exception as e:
#             st.error(f"❌ Prediction failed: {e}")








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

