
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. రియల్ హాస్పిటల్ డేటాని చదవడం
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# 2. కేవలం 4 వివరాలు మాత్రమే తీసుకుంటున్నాం
X = df[['age', 'trestbps', 'chol', 'thalach']]
y = df['target']

# 3. డేటాని విడగొట్టడం (80% చదువుకోవడానికి, 20% ఎగ్జామ్ కి)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. AI మోడల్ కి ట్రైనింగ్ ఇవ్వడం (చదివించడం)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. AI కి ఎగ్జామ్ పెట్టడం & స్కోర్ కనిపెట్టడం
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
accuracy_percentage = round(score * 100, 2)

# వెబ్‌సైట్ డిజైన్ సెట్టింగ్స్
st.set_page_config(page_title="Real Medical AI", page_icon="🩺", layout="centered")

st.title("🩺 రియల్ హార్ట్ డిసీజ్ AI డాక్టర్")

# 📊 మన AI స్కోర్ ని వెబ్‌సైట్ లో చూపించడం
st.info(f"📊 **మన AI డాక్టర్ ఎగ్జామ్ స్కోర్ (Accuracy): {accuracy_percentage}%**")

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
    user_data = pd.DataFrame([[age, bp, chol, hr]], columns=['age', 'trestbps', 'chol', 'thalach'])
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **జాగ్రత్త!** రియల్ హాస్పిటల్ డేటా ప్రకారం మీకు గుండె జబ్బు వచ్చే అవకాశం ఉంది. దయచేసి డాక్టర్‌ని సంప్రదించండి.")
    else:
        st.success("🎉 **అద్భుతం!** రియల్ హాస్పిటల్ డేటా ప్రకారం మీ గుండె చాలా ఆరోగ్యంగా ఉంది. ఇలాగే మెయింటైన్ చేయండి!")
