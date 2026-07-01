import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 1. డమ్మీ హార్ట్ డిసీజ్ డేటాతో మోడల్ ట్రైనింగ్ (రియల్ మోడల్ లాగే పనిచేస్తుంది)
# వయసు, కొలెస్ట్రాల్, బీపీ, హార్ట్ రేట్
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
if st.button("🔍 హెల్త్ రిపోర్ట్ కనిపెట్టు", use_container_width=True):
    # ఇన్పుట్స్ ని స్కేల్ చేయడం (డమ్మీ డేటా కోసం చిన్న అడ్జస్ట్మెంట్)
    user_data = np.array([[age/100, bp/200, chol/400, hr/220]])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **జాగ్రత్త!** మన AI ప్రకారం మీకు గుండె జబ్బు (Heart Disease) వచ్చే అవకాశం ఉంది. దయచేసి డాక్టర్‌ని సంప్రదించండి.")
    else:
        st.success("🎉 **అద్భుతం!** మన AI ప్రకారం మీ గుండె చాలా ఆరోగ్యంగా ఉంది. ఇలాగే మెయింటైన్ చేయండి!")