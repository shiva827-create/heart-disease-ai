import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. రియల్ హాస్పిటల్ డేటాని చదవడం (Pandas మ్యాజిక్)
@st.cache_data
def load_data():
    # ఇది అమెరికాలోని క్లీవ్‌ల్యాండ్ హాస్పిటల్ వాళ్ళ రియల్ డేటాసెట్ లింక్
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# 2. డాక్టర్లు వాడే 4 ముఖ్యమైన వివరాలను మాత్రమే తీసుకుంటున్నాం
# (age: వయసు, trestbps: బీపీ, chol: కొలెస్ట్రాల్, thalach: గుండె వేగం)
X = df[['age', 'trestbps', 'chol', 'thalach']]
y = df['target'] # 1 అంటే జబ్బు ఉన్నట్టు, 0 అంటే లేనట్టు

# 3. AI మోడల్ కి నిజమైన డేటాతో ట్రైనింగ్ ఇవ్వడం
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# వెబ్‌సైట్ డిజైన్ సెట్టింగ్స్
st.set_page_config(page_title="Real Medical AI", page_icon="🩺", layout="centered")

st.title("🩺 రియల్ హార్ట్ డిసీజ్ AI డాక్టర్")
st.write("---")
st.write("### **కింద ఉన్న మీ వివరాలను ఎంటర్ చేసి, గుండె ఆరోగ్యాన్ని చెక్ చేసుకోండి:**")

# యూజర్ ఇన్‌పుట్స్
age = st.number_input("👤 వయసు (Age)", min_value=1, max_value=120, value=45)
bp = st.slider("🩸 రక్తపోటు (Blood Pressure - mmHg)", 80, 200, 120)
chol = st.slider("🧪 కొలెస్ట్రాల్ (Cholesterol - mg/dl)", 100, 400, 200)
hr = st.slider("💓 గరిష్ట గుండె వేగం (Max Heart Rate)", 60, 220, 150)

st.write("---")

# బటన్ అండ్ ప్రిడిక్షన్
if st.button("🔍 నా హెల్త్ రిపోర్ట్ ఇవ్వు", use_container_width=True):
    # యూజర్ ఇచ్చిన డేటాని పాండాస్ (Pandas) ఫార్మాట్ లోకి మార్చడం
    user_data = pd.DataFrame([[age, bp, chol, hr]], columns=['age', 'trestbps', 'chol', 'thalach'])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **జాగ్రత్త!** రియల్ హాస్పిటల్ డేటా ప్రకారం మీకు గుండె జబ్బు వచ్చే అవకాశం ఉంది. దయచేసి డాక్టర్‌ని సంప్రదించండి.")
    else:
        st.success("🎉 **అద్భుతం!** రియల్ హాస్పిటల్ డేటా ప్రకారం మీ గుండె చాలా ఆరోగ్యంగా ఉంది. ఇలాగే మెయింటైన్ చేయండి!")
