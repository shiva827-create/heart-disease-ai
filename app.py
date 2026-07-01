
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("heart.csv.csv")
# 1. గిట్‌హబ్ నుండి నిజమైన పేషెంట్ల డేటాని లోడ్ చేస్తున్నాం
@st.cache_data
def load_real_data():
    # నువ్వు అప్‌లోడ్ చేసిన csv ఫైల్ లింక్ ఇది
    url = "https://raw.githubusercontent.com/shiva827-create/heart-disease-ai/main/heart.csv"
    df = pd.read_csv(url)
    return df

df = load_real_data()

# డేటాని ఇన్‌పుట్స్ (X) మరియు ఆన్సర్స్ (y) గా విడదీస్తున్నాం
X = df[['age', 'bp', 'chol', 'hr']]
y = df['target']

# నిజమైన డేటాతో AI కి ట్రైనింగ్ ఇస్తున్నాం
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# వెబ్‌సైట్ సెట్టింగ్స్
st.set_page_config(page_title="Real Medical AI", page_icon="❤️", layout="centered")

st.title("❤️ ఒరిజినల్ హాస్పిటల్ డేటాసెట్ AI వెబ్‌సైట్")
st.write("---")
st.write("### **నిజమైన పేషెంట్ల డేటా ఆధారంగా మీ గుండె ఆరోగ్యాన్ని చెక్ చేసుకోండి:**")

# యూజర్ ఇన్‌పుట్స్
age = st.number_input("👤 వయసు (Age)", min_value=1, max_value=120, value=55)
bp = st.slider("🩸 రక్తపోటు (Blood Pressure - mmHg)", 80, 200, 140)
chol = st.slider("🧪 కొలెస్ట్రాల్ (Cholesterol - mg/dl)", 100, 400, 250)
hr = st.slider("💓 గరిష్ట గుండె వేగం (Max Heart Rate)", 60, 220, 110)

st.write("---")

# బటన్ అండ్ ప్రిడిక్షన్
if st.button("🔍 నా హెల్త్ రిపోర్ట్ ఇవ్వు", use_container_width=True):
    user_data = np.array([[age, bp, chol, hr]])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **జాగ్రత్త!** ఒరిజినల్ హాస్పిటల్ డేటా ప్రకారం మీకు గుండె జబ్బు (Heart Disease) వచ్చే అవకాశం చాలా ఎక్కువగా ఉంది. దయచేసి వెంటనే డాక్టర్‌ని సంప్రదించండి.")
    else:
        st.success("🎉 **అద్భుతం!** ఒరిజినల్ హాస్పిటల్ డేటా ప్రకారం మీ గుండె చాలా ఆరోగ్యంగా ఉంది. ఇలాగే మెయిنتైన్ చేయండి!")
