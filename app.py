
# # import streamlit as st
# # import numpy as np
# # from joblib import load
# # import os

# # # ================== PAGE CONFIG ==================
# # st.set_page_config(
# #     page_title="CardioSense | AI Heart Disease Prediction",
# #     page_icon="❤️",
# #     layout="wide"
# # )

# # # ================== LOAD MODEL ==================
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # model = load(os.path.join(BASE_DIR, "model", "rf_model.joblib"))
# # scaler = load(os.path.join(BASE_DIR, "model", "scaler.joblib"))

# # # ================== SESSION STATE ==================
# # if "active_menu" not in st.session_state:
# #     st.session_state.active_menu = "Home"

# # def set_menu(menu_name):
# #     st.session_state.active_menu = menu_name

# # # ================== CUSTOM CSS ==================
# # st.markdown("""
# # <style>
# # /* Main background */
# # .main {
# #     background-color: #f5f7fb;
# # }

# # /* Cards */
# # .card {
# #     background-color: #ffffff;
# #     padding: 30px;
# #     border-radius: 16px;
# #     box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
# #     margin-bottom: 25px;
# # }

# # /* Titles */
# # .page-title {
# #     color: #2b2d42;
# #     font-size: 36px;
# #     font-weight: 700;
# # }

# # /* Buttons */
# # .stButton > button {
# #     background-color: #8b0000;
# #     color: white;
# #     font-size: 18px;
# #     padding: 12px 30px;
# #     border-radius: 12px;
# # }
# # </style>
# # """, unsafe_allow_html=True)

# # # ================== SIDEBAR ==================
# # st.sidebar.markdown(
# #     """
# #     <div style="padding:15px; background-color:#2b2d42; border-radius:10px; margin-bottom:20px;">
# #         <h2 style="color:white; margin-bottom:5px;">❤️ CardioSense</h2>
# #         <p style="color:#f1f1f1; font-size:14px;">AI-Powered Cardiovascular Risk Assessment System</p>
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )

# # menu_items = [
# #     ("🏠 Home", "Home"),
# #     ("🩺 Heart Disease Prediction", "Prediction"),
# #     ("📊 Machine Learning Algorithms", "Algorithms"),
# #     ("📈 Model Performance & Evaluation", "Performance"),
# #     ("⚠ Medical Disclaimer", "Disclaimer"),
# #     ("ℹ About Project", "About")
# # ]

# # # Sidebar buttons (working)
# # for label, name in menu_items:
# #     if st.session_state.active_menu == name:
# #         st.sidebar.button(label, key=name, on_click=set_menu, args=(name,), help="Active", use_container_width=True)
# #     else:
# #         st.sidebar.button(label, key=name, on_click=set_menu, args=(name,), use_container_width=True)

# # st.sidebar.markdown("---")
# # st.sidebar.markdown(
# #     """
# #     **Why CardioSense?**  
# #     - Early risk identification  
# #     - Data-driven medical insights  
# #     - Designed for learning & awareness  
# #     """
# # )

# # # ================== PAGES ==================
# # if st.session_state.active_menu == "Home":
# #     st.markdown("<div class='page-title'>AI-Based Heart Disease Prediction</div>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class="card">
# #     <h3>Overview</h3>
# #     <p>
# #     Cardiovascular diseases (CVDs) are the leading cause of death globally.
# #     CardioSense uses <b>machine learning</b> to predict heart disease risk.
# #     </p>
# #     </div>
# #     """, unsafe_allow_html=True)

# # elif st.session_state.active_menu == "Prediction":
# #     st.markdown("<div class='page-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)

# #     col1, col2, col3 = st.columns(3)

# #     with col1:
# #         age = st.number_input("Age (years)", 20, 100, 45)
# #         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# #         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
# #         thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

# #     with col2:
# #         sex = st.selectbox("Sex", ["Male", "Female"])
# #         cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
# #         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
# #         restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])

# #     with col3:
# #         exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
# #         oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
# #         slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
# #         ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
# #         thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

# #     sex = 1 if sex == "Male" else 0
# #     fbs = 1 if fbs == "Yes" else 0
# #     exang = 1 if exang == "Yes" else 0

# #     if st.button("Predict Heart Disease Risk"):
# #         input_data = np.array([[age, sex, cp, trestbps, chol,
# #                                 fbs, restecg, thalach,
# #                                 exang, oldpeak, slope, ca, thal]])
# #         input_scaled = scaler.transform(input_data)
# #         result = model.predict(input_scaled)
# #         if result[0] == 1:
# #             st.error("🔴 High Risk of Cardiovascular Disease detected.")
# #         else:
# #             st.success("🟢 Low Risk of Cardiovascular Disease detected.")

# # elif st.session_state.active_menu == "Algorithms":
# #     st.markdown("<div class='page-title'>Algorithms & Methodology</div>", unsafe_allow_html=True)

# # elif st.session_state.active_menu == "Performance":
# #     st.markdown("<div class='page-title'>Model Evaluation</div>", unsafe_allow_html=True)

# # elif st.session_state.active_menu == "Disclaimer":
# #     st.markdown("<div class='page-title'>Medical Disclaimer</div>", unsafe_allow_html=True)

# # elif st.session_state.active_menu == "About":
# #     st.markdown("<div class='page-title'>About CardioSense</div>", unsafe_allow_html=True)
# # # --------------- HOME ----------------
# # if st.session_state.active_menu == "Home":
# #     st.markdown("<div class='page-title'>AI-Based Heart Disease Prediction</div>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class="card">
# #     <h3>Overview</h3>
# #     <p>
# #     Cardiovascular diseases (CVDs) are the leading cause of death globally.
# #     CardioSense is an intelligent web application that uses
# #     <b>machine learning techniques</b> to predict whether a person is at
# #     <b>high or low risk</b> of heart disease based on clinical parameters.
# #     </p>

# #     <h4>How It Works</h4>
# #     <p>
# #     The system analyzes patient inputs such as age, blood pressure,
# #     cholesterol level, ECG results, heart rate, and other medical factors.
# #     These inputs are processed through a trained Random Forest model to
# #     generate a reliable prediction.
# #     </p>

# #     <h4>Purpose</h4>
# #     <ul>
# #         <li>Educational demonstration of AI in healthcare</li>
# #         <li>Understanding risk factors of heart disease</li>
# #         <li>Supporting early awareness and prevention</li>
# #     </ul>
# #     </div>
# #     """, unsafe_allow_html=True)

# # # --------------- HEART DISEASE PREDICTION ----------------
# # elif st.session_state.active_menu == "Prediction":
# #     st.markdown("<div class='page-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class="card">
# #     <p>
# #     Please enter the patient's clinical details carefully.
# #     All values are based on standard medical diagnostic parameters.
# #     </p>
# #     </div>
# #     """, unsafe_allow_html=True)

# #     col1, col2, col3 = st.columns(3)

# #     with col1:
# #         age = st.number_input("Age (years)", 20, 100, 45)
# #         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# #         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
# #         thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

# #     with col2:
# #         sex = st.selectbox("Sex", ["Male", "Female"])
# #         cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
# #         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
# #         restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])

# #     with col3:
# #         exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
# #         oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
# #         slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
# #         ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
# #         thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

# #     sex = 1 if sex == "Male" else 0
# #     fbs = 1 if fbs == "Yes" else 0
# #     exang = 1 if exang == "Yes" else 0

# #     if st.button("Predict Heart Disease Risk"):
# #         input_data = np.array([[age, sex, cp, trestbps, chol,
# #                                 fbs, restecg, thalach,
# #                                 exang, oldpeak, slope, ca, thal]])

# #         input_scaled = scaler.transform(input_data)
# #         result = model.predict(input_scaled)

# #         if result[0] == 1:
# #             st.error("🔴 High Risk of Cardiovascular Disease detected.")
# #         else:
# #             st.success("🟢 Low Risk of Cardiovascular Disease detected.")

# # # --------------- ALGORITHMS ----------------
# # elif st.session_state.active_menu == "Algorithms":
# #     st.markdown("<div class='page-title'>Algorithms & Methodology</div>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class="card">
# #     <h3>Random Forest Classifier</h3>
# #     <p>
# #     Random Forest is an ensemble learning technique that builds multiple
# #     decision trees and combines their outputs to improve prediction accuracy
# #     and reduce overfitting.
# #     </p>

# #     <h4>Why Random Forest?</h4>
# #     <ul>
# #         <li>Handles complex non-linear relationships</li>
# #         <li>Robust to noise and missing values</li>
# #         <li>High accuracy for medical datasets</li>
# #     </ul>
# #     </div>
# #     """, unsafe_allow_html=True)

# # # --------------- PERFORMANCE ----------------
# # elif st.session_state.active_menu == "Performance":
# #     st.markdown("<div class='page-title'>Model Evaluation</div>", unsafe_allow_html=True)
# #     st.image("assets/confusion_matrix.png", caption="Confusion Matrix")
# #     st.image("assets/model_comparison.png", caption="Model Accuracy Comparison")

# # # --------------- DISCLAIMER ----------------
# # elif st.session_state.active_menu == "Disclaimer":
# #     st.markdown("""
# #     <div class="card">
# #     <h3>Medical Disclaimer</h3>
# #     <p>
# #     This application is intended strictly for <b>educational and academic purposes</b>.
# #     The predictions generated by CardioSense should not be considered as a medical diagnosis.
# #     Always consult a qualified healthcare professional for medical advice.
# #     </p>
# #     </div>
# #     """, unsafe_allow_html=True)

# # # --------------- ABOUT ----------------
# # elif st.session_state.active_menu == "About":
# #     st.markdown("""
# #     <div class="card">
# #     <h3>About CardioSense</h3>
# #     <p>
# #     CardioSense is a final-year machine learning project demonstrating
# #     the application of artificial intelligence in healthcare diagnostics.
# #     </p>

# #     <b>Author:</b> Gohil Arjunsinh  
# #     <br><b>Domain:</b> Machine Learning & Healthcare  
# #     <br><b>Technology:</b> Python, Streamlit, Scikit-learn
# #     </div>
# #     """, unsafe_allow_html=True)



# import streamlit as st
# import numpy as np
# from joblib import load
# import os

# # ================== PAGE CONFIG ==================
# st.set_page_config(
#     page_title="CardioSense | AI Heart Disease Prediction",
#     page_icon="❤️",
#     layout="wide"
# )
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "model", "lr_model.joblib")

# st.write("Model path:", MODEL_PATH)
# st.write("Model size:", os.path.getsize(MODEL_PATH))

# model = load(MODEL_PATH)

# # ================== LOAD MODEL ==================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = load(os.path.join(BASE_DIR, "model", "lr_model.joblib"))
# scaler = load(os.path.join(BASE_DIR, "model", "scaler.joblib"))

# # ================== SESSION STATE ==================
# if "active_menu" not in st.session_state:
#     st.session_state.active_menu = "Home"

# def set_menu(menu_name):
#     st.session_state.active_menu = menu_name

# # ================== CUSTOM CSS ==================
# st.markdown("""
# <style>

# /* Main background */
# .main {
#     background-color: #f5f7fb;
# }

# /* Top menu bar */
# .topbar {
#     background-color: #ffffff;
#     padding: 15px 30px;
#     border-bottom: 1px solid #e5e7eb;
#     margin-bottom: 25px;
# }
# .topbar-title {
#     font-size: 26px;
#     font-weight: 700;
#     color: #2b2d42;
# }
# .topbar-subtitle {
#     font-size: 14px;
#     color: #6b7280;
# }

# /* Cards */
# .card {
#     background-color: #ffffff;
#     padding: 30px;
#     border-radius: 16px;
#     box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
#     margin-bottom: 25px;
# }

# /* Titles */
# .page-title {
#     color: #2b2d42;
#     font-size: 34px;
#     font-weight: 700;
# }

# /* Buttons */
# .stButton > button {
#     background-color: #2563eb;
#     color: white;
#     font-size: 16px;
#     padding: 10px 24px;
#     border-radius: 10px;
# }

# </style>
# """, unsafe_allow_html=True)


# # ================== TOP MENU BUTTONS ==================
# menu_cols = st.columns([1.4, 1.8, 1.8, 2.4, 1.8, 1.4])

# menu_map = [
#     ("Home", "Home"),
#     ("Prediction", "Prediction"),
#     ("Algorithms", "Algorithms"),
#     ("Model Performance", "Performance"),
#     ("Disclaimer", "Disclaimer"),
#     ("About", "About"),
# ]

# for col, (label, key) in zip(menu_cols, menu_map):
#     with col:
#         if st.button(label, use_container_width=True):
#             set_menu(key)



# # ================== TOP MENU BAR ==================
# st.markdown("""
# <div class="topbar">
#     <div class="topbar-title">❤️ CardioSense</div>
#     <div class="topbar-subtitle">
#         AI-Powered Cardiovascular Risk Assessment System
#     </div>
# </div>
# """, unsafe_allow_html=True)


# # ================== SIDEBAR ==================
# st.sidebar.markdown(
#     """
#     <div style="padding:15px; background-color:#2b2d42; border-radius:10px; margin-bottom:20px;">
#         <h2 style="color:white; margin-bottom:5px;">❤️ CardioSense</h2>
#         <p style="color:#f1f1f1; font-size:14px;">
#         AI-Powered Cardiovascular Risk Assessment System
#         </p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# menu_items = [
#     ("🏠 Home", "Home"),
#     ("🩺 Heart Disease Prediction", "Prediction"),
#     ("📊 Machine Learning Algorithms", "Algorithms"),
#     ("📈 Model Performance & Evaluation", "Performance"),
#     ("⚠ Medical Disclaimer", "Disclaimer"),
#     ("ℹ About Project", "About")
# ]

# for label, name in menu_items:
#     st.sidebar.button(
#         label,
#         key=f"side_{name}",
#         on_click=set_menu,
#         args=(name,),
#         use_container_width=True
#     )

# st.sidebar.markdown("---")
# st.sidebar.markdown(
#     """
#     **Why CardioSense?**  
#     - Early risk identification  
#     - Data-driven medical insights  
#     - Designed for learning & awareness  
#     """
# )

# # ================== PAGES ==================

# # -------- HOME --------
# if st.session_state.active_menu == "Home":
#     st.markdown("<div class='page-title'>AI-Based Heart Disease Prediction</div>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class="card">
#     <h3>Overview</h3>
#     <p>
#     Cardiovascular diseases (CVDs) are the leading cause of death globally.
#     CardioSense is an intelligent web application that uses
#     <b>machine learning techniques</b> to assess heart disease risk
#     using clinical parameters.  
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

# # -------- PREDICTION --------
# elif st.session_state.active_menu == "Prediction":
#     st.markdown("<div class='page-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.number_input("Age (years)", 20, 100, 45)
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
#         thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

#     with col2:
#         sex = st.selectbox("Sex", ["Male", "Female"])
#         cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
#         restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])

#     with col3:
#         exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
#         oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
#         slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
#         ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
#         thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

#     sex = 1 if sex == "Male" else 0
#     fbs = 1 if fbs == "Yes" else 0
#     exang = 1 if exang == "Yes" else 0

#     if st.button("Predict Heart Disease Risk"):
#         input_data = np.array([[age, sex, cp, trestbps, chol,
#                                 fbs, restecg, thalach,
#                                 exang, oldpeak, slope, ca, thal]])
#         input_scaled = scaler.transform(input_data)
#         result = model.predict(input_scaled)

#         if result[0] == 1:
#             st.error("🔴 High Risk of Cardiovascular Disease detected.")
#         else:
#             st.success("🟢 Low Risk of Cardiovascular Disease detected.")

# # -------- ALGORITHMS --------
# elif st.session_state.active_menu == "Algorithms":
#     st.markdown("<div class='page-title'>Machine Learning Algorithms</div>", unsafe_allow_html=True)

#     st.markdown("""
#     <div class="card">
#     <h3>Algorithm Used: Random Forest Classifier</h3>

#     <p>
#     This project uses the <b>Random Forest</b> algorithm, an ensemble learning
#     technique that combines multiple decision trees to improve prediction accuracy
#     and reduce overfitting. It is well-suited for medical datasets due to its
#     robustness and ability to handle non-linear relationships.
#     </p>

#     <h4>Why Random Forest?</h4>
#     <ul>
#         <li>High accuracy on structured medical data</li>
#         <li>Handles missing and noisy values effectively</li>
#         <li>Reduces overfitting compared to single decision trees</li>
#         <li>Provides feature importance insights</li>
#     </ul>

#     <h4>Algorithms Compared</h4>
#     <p>
#     During experimentation, multiple machine learning algorithms were evaluated:
#     </p>
#     <ul>
#         <li>Logistic Regression</li>
#         <li>K-Nearest Neighbors (KNN)</li>
#         <li>Support Vector Machine (SVM)</li>
#         <li>Decision Tree</li>
#         <li><b>Random Forest (Final Model)</b></li>
#     </ul>

#     <p>
#     Based on accuracy and stability, Random Forest outperformed other models
#     and was selected for final deployment.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

# # -------- PERFORMANCE --------
# elif st.session_state.active_menu == "Performance":
#     st.markdown("<div class='page-title'>Model Performance & Evaluation</div>", unsafe_allow_html=True)

#     st.markdown("""
#     <div class="card">
#     <h3>Model Evaluation Metrics</h3>
#     <p>
#     The performance of the heart disease prediction model was evaluated using
#     standard metrics such as accuracy, precision, recall, and F1-score.
#     These metrics ensure the reliability of predictions on unseen data.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#     <div class="card">
#     <h3>Confusion Matrix</h3>
#     <p>
#     The confusion matrix visually represents the model’s classification results.
#     It helps analyze correct and incorrect predictions made by the model.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.image(
#         os.path.join(r"C:\Users\Owner\Downloads\GOHILRAJ_MLDL_PROJECT\Cardio_fronted\assets", "confusion_matrix.png"),
#         caption="Confusion Matrix",
#         width=700
#     )

#     st.markdown("""
#     <div class="card">
#     <h3>Algorithm Comparison</h3>
#     <p>
#     This graph compares the accuracy of multiple machine learning algorithms.
#     Random Forest achieved the best performance and was selected as the final model.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.image(
#         os.path.join(BASE_DIR, "assets", "model_comparison.png"),
#         caption="Model Accuracy Comparison",
#         use_container_width=True
#     )

# # -------- DISCLAIMER --------
# elif st.session_state.active_menu == "Disclaimer":
#     st.markdown("""
#     <div class="card">
#     <h3>Medical Disclaimer</h3>
#     <p>
#     This application is strictly for educational purposes
#     and should not be used for real medical diagnosis.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

# # -------- ABOUT --------
# elif st.session_state.active_menu == "About":
#     st.markdown("<div class='page-title'>About This Project</div>", unsafe_allow_html=True)

#     st.markdown("""
#     <div class="card">
#     <h3>Project Overview</h3>
#     <p>
#     CardioSense is a college-level machine learning project developed to demonstrate
#     the application of artificial intelligence in healthcare diagnostics.
#     The system predicts the risk of heart disease based on patient clinical data.
#     </p>

#     <h3>Dataset</h3>
#     <p>
#     The dataset used contains clinical attributes such as age, blood pressure,
#     cholesterol levels, ECG results, heart rate, and exercise-induced parameters.
#     These features are commonly used by cardiologists for diagnosis.
#     </p>

#     <h3>Project Workflow</h3>
#     <ol>
#         <li>Data collection and preprocessing</li>
#         <li>Feature scaling using RobustScaler</li>
#         <li>Model training using multiple ML algorithms</li>
#         <li>Model evaluation and selection</li>
#         <li>Deployment using Streamlit</li>
#     </ol>

#     <h3>Learning Outcome</h3>
#     <ul>
#         <li>Hands-on experience with machine learning models</li>
#         <li>Understanding healthcare data analysis</li>
#         <li>Frontend-backend integration using Streamlit</li>
#         <li>Real-world AI project implementation</li>
#     </ul>

#     <p>
#     This project is developed strictly for educational purposes and serves as
#     a foundation for understanding AI-driven medical decision support systems.
#     </p>

#     <b>Developed By:</b> Gohil Arjunsinh  
#     <br><b>Domain:</b> Machine Learning & Healthcare  
#     <br><b>Project Type:</b> College Final Year Project
#     </div>
#     """, unsafe_allow_html=True)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="CardioSense | AI Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ================== BASE DIR ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ================== LOAD MODEL PARAMETERS ==================
coef = np.load(os.path.join(MODEL_DIR, "lr_coef.npy"))
intercept = np.load(os.path.join(MODEL_DIR, "lr_intercept.npy"))
center = np.load(os.path.join(MODEL_DIR, "scaler_center.npy"))
scale = np.load(os.path.join(MODEL_DIR, "scaler_scale.npy"))

# ================== HELPER FUNCTIONS ==================
def scale_input(X):
    return (X - center) / scale

def predict_lr(X):
    z = np.dot(X, coef.T) + intercept
    prob = 1 / (1 + np.exp(-z))
    pred = (prob >= 0.5).astype(int)
    return pred, prob

# ================== SESSION STATE ==================
if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Home"

def set_menu(menu):
    st.session_state.active_menu = menu

# ================== CUSTOM CSS ==================
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

<style>
/* ---------- HERO SECTION ---------- */
.hero {
    background-image: linear-gradient(
        rgba(30, 58, 138, 0.85),
        rgba(37, 99, 235, 0.85)
    ),
    url("Heart.png");
    background-size: cover;
    background-position: center;
    padding: 80px 60px;
    border-radius: 24px;
    color: white;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 15px;
}

.hero-subtitle {
    font-size: 20px;
    max-width: 700px;
    line-height: 1.6;
    opacity: 0.95;
}

/* KPI override for home */
.kpi {
    background: white;
    color: #1e293b;
    box-shadow: 0 15px 35px rgba(0,0,0,0.12);
}

.main { background-color: #f5f7fb; }

/* Top Bar */
.topbar {
    background-color: #ffffff;
    padding: 18px 30px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 25px;
}
.topbar-title {
    font-size: 26px;
    font-weight: 700;
    color: #2b2d42;
}
.topbar-subtitle {
    font-size: 14px;
    color: #6b7280;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Titles */
.page-title {
    color: #2b2d42;
    font-size: 34px;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================== TOP MENU BUTTONS ==================
menu_cols = st.columns([1.4, 1.8, 1.8, 2.4, 1.8, 1.4])

menu_map = [
    ("Home", "Home"),
    ("Prediction", "Prediction"),
    ("Algorithms", "Algorithms"),
    ("Model Performance", "Performance"),
    ("Disclaimer", "Disclaimer"),
    ("About", "About"),
]

for col, (label, key) in zip(menu_cols, menu_map):
    with col:
        if st.button(label, use_container_width=True):
            set_menu(key)


# ================== TOP BAR ==================
st.markdown("""
<div class="topbar">
    <div class="topbar-title">❤️ CardioSense</div>
    <div class="topbar-subtitle">
        AI-Powered Cardiovascular Risk Assessment System
    </div>
</div>
""", unsafe_allow_html=True)


# ================== SIDEBAR ==================
st.sidebar.markdown("""
<div style="padding:15px; background-color:#2b2d42; border-radius:12px; margin-bottom:20px;">
    <h2 style="color:white; margin-bottom:5px;">❤️ CardioSense</h2>
    <p style="color:#f1f1f1; font-size:14px;">
    Logistic Regression Based Heart Disease Prediction
    </p>
</div>
""", unsafe_allow_html=True)

menu_items = [
    ("🏠 Home", "Home"),
    ("🩺 Prediction", "Prediction"),
    ("📊 Algorithms", "Algorithms"),
    ("📈 Performance", "Performance"),
    ("⚠ Disclaimer", "Disclaimer"),
    ("ℹ About", "About")
]

for label, name in menu_items:
    st.sidebar.button(label, on_click=set_menu, args=(name,), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Why CardioSense?**
- Early disease risk detection  
- Data-driven healthcare insights  
- Educational & academic use  
""")

# ================== HOME ==================
if st.session_state.active_menu == "Home":

    # ---------- HERO SECTION ----------
    st.markdown("""
    <div class="hero">
        <div class="hero-title">❤️ CardioSense</div>
        <div class="hero-subtitle">
            An AI-powered heart disease risk prediction system that leverages
            machine learning to assist in early cardiovascular risk assessment
            using clinical patient data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- KPI SECTION ----------
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown("""
        <div class='kpi'>
            🧠 <b>ML Algorithm</b><br>
            Logistic Regression
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class='kpi'>
            📊 <b>Prediction Accuracy</b><br>
            72.73%
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class='kpi'>
            🏥 <b>Clinical Features</b><br>
            13 Parameters
        </div>
        """, unsafe_allow_html=True)

    # ---------- ABOUT CARD ----------
    st.markdown("""
    <div class="card">
    <h2>Why Choose CardioSense?</h2>

    <p>
    Cardiovascular diseases remain one of the leading causes of mortality worldwide.
    CardioSense is designed as an academic and research-oriented system that demonstrates
    how artificial intelligence can support early heart disease risk assessment.
    </p>

    <ul>
        <li>✔ Early identification of cardiovascular risk</li>
        <li>✔ AI-driven insights from clinical data</li>
        <li>✔ Interpretable and transparent ML model</li>
        <li>✔ Designed for educational & academic use</li>
        <li>✔ Lightweight and fast prediction system</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ================== PREDICTION ==================
elif st.session_state.active_menu == "Prediction":
    st.markdown("<div class='page-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    Please enter patient clinical details carefully.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", 20, 100, 45)
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])

    with col3:
        exang = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease Risk"):
        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])

        input_scaled = scale_input(input_data)
        pred, prob = predict_lr(input_scaled)
        if pred[0][0] == 1:
            st.markdown(f"""
            <div class="result-box high-risk">
            🔴 High Risk of Heart Disease<br>
             Probability: {prob[0][0]*100:.2f}%
             </div>
             """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
            <div class="result-box low-risk">
            🟢 Low Risk of Heart Disease<br>
            Confidence: {(1-prob[0][0])*100:.2f}%
            </div>
            """, unsafe_allow_html=True)


# ================== ALGORITHMS ==================
elif st.session_state.active_menu == "Algorithms":

    st.markdown("<div class='page-title'>Machine Learning Algorithm</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Logistic Regression</h3>

    <p>
    Logistic Regression is a supervised machine learning algorithm commonly used
    for binary classification problems, particularly in healthcare applications.
    In this project, it is employed to predict the risk of heart disease
    (<b>Presence / Absence</b>) based on clinical parameters.
    </p>

    <h4>Why Logistic Regression?</h4>
    <ul>
        <li>
            <b>Best Performing Model:</b> Achieved the highest test accuracy
            of <b>72.73%</b>, outperforming Random Forest (<b>72.11%</b>)
        </li>
        <li>
            <b>Interpretability:</b> Model coefficients clearly indicate how
            each clinical feature influences heart disease risk
        </li>
        <li>
            <b>Medical Suitability:</b> Transparent decision-making is essential
            in healthcare and clinical decision support systems
        </li>
        <li>
            <b>Stable Generalization:</b> Similar accuracy across training,
            testing, and cross-validation indicates minimal overfitting
        </li>
    </ul>

    <h4>How the Model Works (Intuition)</h4>
    <p>
    Logistic Regression computes a weighted sum of the input clinical features
    and applies a sigmoid activation function to convert this value into
    a probability score between 0 and 1.
    </p>

    <p style="text-align:center; font-weight:bold;">
    P(Heart Disease) = 1 / (1 + e<sup>-z</sup>)
    </p>

    <p>
    A probability greater than 0.5 is classified as <b>High Risk</b>,
    while values below 0.5 indicate <b>Low Risk</b>.
    </p>

    <h4>Algorithms Evaluated During Experimentation</h4>
    <ul>
        <li>
            <b>Logistic Regression:</b> Selected final model due to higher
            accuracy, stability, and interpretability
        </li>
        <li>
            <b>Random Forest:</b> Achieved slightly lower accuracy
            (72.11%) with higher computational complexity
        </li>
    </ul>

    <h4>Final Model Selection Rationale</h4>
    <p>
    Although Random Forest is a powerful ensemble learning technique,
    Logistic Regression was chosen as the final model because it delivered
    marginally higher accuracy on this dataset and provides clear,
    explainable predictions—an important requirement in medical
    decision-support applications.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ================== PERFORMANCE ==================
elif st.session_state.active_menu == "Performance":

    st.markdown("<div class='page-title'>Model Performance & Evaluation</div>", unsafe_allow_html=True)

    # ================= KPI METRICS =================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("<div class='kpi'>🎯 Accuracy<br><b>72.73%</b></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='kpi'>📊 Cross Validation<br><b>72.03%</b></div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='kpi'>🧪 Training Accuracy<br><b>72.72%</b></div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='kpi'>🧠 Testing Accuracy<br><b>72.72%</b></div>", unsafe_allow_html=True)

    # ================= CONFUSION MATRIX =================
    st.markdown("""
    <div class="card">
    <h3>Confusion Matrix Analysis</h3>
    <p>
    The confusion matrix provides a detailed breakdown of correct and incorrect
    predictions made by the Logistic Regression model.
    </p>

    <ul>
        <li><b>True Positives:</b> Correctly identified heart disease cases</li>
        <li><b>True Negatives:</b> Correctly identified healthy patients</li>
        <li><b>False Positives:</b> Healthy patients predicted as diseased</li>
        <li><b>False Negatives:</b> Diseased patients predicted as healthy</li>
    </ul>

    <p>
    In medical diagnosis, minimizing <b>false negatives</b> is critical,
    as missing a heart disease case can be life-threatening.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        os.path.join(BASE_DIR, "assets", "confusion_matrix.png"),
        width=650
    )

    # ================= ALGORITHM COMPARISON =================
    st.markdown("""
    <div class="card">
    <h3>Algorithm Comparison</h3>

    <p>
    Multiple machine learning algorithms were evaluated on the same dataset.
    Although Random Forest achieved similar accuracy, Logistic Regression
    was selected for final deployment.
    </p>

    <table style="width:100%; border-collapse:collapse;">
        <tr>
            <th align="left">Algorithm</th>
            <th align="left">Accuracy</th>
            <th align="left">Remarks</th>
        </tr>
        <tr>
            <td>Logistic Regression</td>
            <td>72.73%</td>
            <td>Interpretable, stable, medical-friendly</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>72.11%</td>
            <td>Higher complexity, less interpretable</td>
        </tr>
    </table>

    <p style="margin-top:15px;">
    Logistic Regression was chosen due to its transparency,
    explainability, and suitability for clinical decision support systems.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        os.path.join(BASE_DIR, "assets", "model_comparison.png"),
        width=600
    )
    st.markdown("""
    <div class="card">
    <h3>Feature Importance (Logistic Regression)</h3>
    <p>
    Positive coefficients increase heart disease risk,
    while negative values decrease risk.
    </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
        "Age", "Sex", "Chest Pain", "BP", "Cholesterol",
        "FBS", "Rest ECG", "Max HR", "Exercise Angina",
        "Oldpeak", "Slope", "Major Vessels", "Thal"
    ]

    coefficients = coef.flatten()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(features, coefficients)
    ax.set_xlabel("Coefficient Value")

    st.pyplot(fig)


    # ================= FINAL INTERPRETATION =================
    st.markdown("""
    <div class="card">
    <h3>Final Interpretation</h3>

    <ul>
        <li>The model demonstrates consistent performance across training, testing,
            and cross-validation.</li>
        <li>Accuracy of ~73% is acceptable for an academic medical AI system.</li>
        <li>Logistic Regression provides interpretable coefficients,
            helping understand clinical risk factors.</li>
        <li>The model is suitable for <b>educational and research purposes</b>,
            not clinical diagnosis.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ================== DISCLAIMER ==================
elif st.session_state.active_menu == "Disclaimer":
    st.markdown("""
    <div class="card">
    <h3>Medical & Legal Disclaimer</h3>

    - The application demonstrates the use of **Machine Learning**
      techniques in the healthcare domain for educational and research purposes only.

    - The predictions generated by this system are based on patterns learned from
      historical clinical data and **do not represent a medical diagnosis**.

    - This application should **not be used** as a substitute for professional
      medical advice, diagnosis, or treatment.

    - The developers do **not claim clinical accuracy** or real-world reliability
      and are not responsible for decisions made using this system.

    - Any health-related decisions must always be taken in consultation with
      licensed medical professionals such as doctors or cardiologists.

    - This system is intended solely for **learning, experimentation, and academic demonstration**
      of machine learning concepts in healthcare analytics.
    </div>
    """, unsafe_allow_html=True)

# ================== ABOUT ==================
elif st.session_state.active_menu == "About":
    st.markdown("<div class='page-title'>About This Project</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>Project Overview</h3>
    <p>
    CardioSense is a college-level machine learning project developed to demonstrate
    the application of artificial intelligence in healthcare diagnostics.
    The system predicts the risk of heart disease based on patient clinical data.
    </p>

    <h3>Dataset</h3>
    <p>
    The dataset used contains clinical attributes such as age, blood pressure,
    cholesterol levels, ECG results, heart rate, and exercise-induced parameters.
    These features are commonly used by cardiologists for diagnosis.
    </p>

    <h3>Project Workflow</h3>
    <ol>
        <li>Data collection and preprocessing</li>
        <li>Feature scaling using RobustScaler</li>
        <li>Model training using multiple ML algorithms</li>
        <li>Model evaluation and selection</li>
        <li>Deployment using Streamlit</li>
    </ol>

    <h3>Learning Outcome</h3>
    <ul>
        <li>Hands-on experience with machine learning models</li>
        <li>Understanding healthcare data analysis</li>
        <li>Frontend-backend integration using Streamlit</li>
        <li>Real-world AI project implementation</li>
    </ul>

    <p>
    This project is developed strictly for educational purposes and serves as
    a foundation for understanding AI-driven medical decision support systems.
    </p>

    <b>Developed By:</b> Gohil Arjunsinh  
    <br><b>Domain:</b> Machine Learning & Healthcare  
    </div>
    """, unsafe_allow_html=True)