
import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 1. మోడల్ ట్రైనింగ్ (ఈసారి పక్కా రూల్స్ ఉండేలా సెట్ చేశాం)
X, y = make_classification(n_samples=200, n_features=4, n_informative=4, 
                           n_redundant=0, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# వెబ్‌సైట్ సెట్టింగ్స్
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ గుండె జబ్బు గుర్తింపు AI వెబ్‌సైట్")
st.write("---")
st.write("### **కింద ఉన్న వివరాలను ఎంటర్ చేసి, గుండె ఆరోగ్యాన్ని చెక్ చేసుకోండి:**")

# యూజర్ ఇన్‌పుట్స్
age = st.number_input("👤 వయసు (Age)", min_value=1, max_value=120, value=45)
bp = st.slider("🩸 రక్తపోటు (Blood Pressure - mmHg)", 80, 200, 120)
chol = st.slider("🧪 కొలెస్ట్రాల్ (Cholesterol - mg/dl)", 100, 400, 200)
hr = st.slider("💓 గరిష్ట గుండె వేగం (Max Heart Rate)", 60, 220, 150)

st.write("---")

# బటన్ అండ్ ప్రిడిక్షన్
if st.button("🔍 ஹெல்த் రిపోర్ట్ కనిపెట్టు", use_container_width=True):
    # ఎటువంటి కన్ఫ్యూజన్ లేకుండా నేరుగా వాల్యూస్ ఇస్తున్నాం
    user_data = np.array([[age, bp, chol, hr]])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **జాగ్రత్త!** మన AI ప్రకారం మీకు గుండె జబ్బు (Heart Disease) వచ్చే అవకాశం ఉంది. దయచేసి డాక్టర్‌ని సంప్రదించండి.")
    else:
        st.success("🎉 **అద్భుతం!** మన AI ప్రకారం మీ గుండె చాలా ఆరోగ్యంగా ఉంది. ఇలా گے మెయింటైన్ చేయండి!")
