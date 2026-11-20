import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Text cleaning function for the model
def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ة', 'ه', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Bilingual Titles (No reshaper needed!)
st.title("Arabic Medical Category Predictor")
st.subheader("التنبؤ بالفئة الطبية للنص العربي")

# Bilingual Input Label
user_input = st.text_area("Enter your medical question (in Arabic):\nأدخل سؤالك الطبي باللغة العربية:")

if st.button("Predict / توقع"):
    if user_input:
        # --- INNOVATION: MEDICAL KEYWORD EXTRACTION ---
        keywords = ["ألم", "صداع", "قلب", "سخونة", "حرارة", "بطن", "سعال", "تعب", "دوخة"]
        keyword_pattern = "|".join(keywords)
        detected_keywords = re.findall(keyword_pattern, user_input)
        
        if detected_keywords:
            unique_keywords = list(set(detected_keywords))
            st.info(f"Detected Medical Keywords / الكلمات الطبية المكتشفة: {unique_keywords}")
        # ----------------------------------------------

        # Clean and Predict
        cleaned = clean_arabic_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)
        
        # Display Output
        st.success(f"Predicted Category / الفئة المتوقعة: {prediction[0]}")
        
    else:
        st.warning("Please enter text. / الرجاء إدخال نص.")
