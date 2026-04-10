import streamlit as st
import tensorflow as tf
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- SESSION STATE INIT ---
if 'user_input_val' not in st.session_state:
    st.session_state['user_input_val'] = ""

# --- CACHING RESOURCES ---
@st.cache_resource
def load_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    model = tf.keras.models.load_model("spam_classifier.keras")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

# Load model and vectorizer
model, tfidf = load_resources()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    # Remove special characters but keep spaces
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text]).toarray()
    pred = model.predict(vector)[0][0]
    return float(pred)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        background-color: #1e2130;
        color: white;
        border-radius: 10px;
        border: 1px solid #4a5568;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #3182ce;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4299e1;
        border-color: #4299e1;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #4a5568;
    }
    .spam-label {
        color: #f56565;
        font-size: 24px;
        font-weight: bold;
    }
    .ham-label {
        color: #48bb78;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("assets/spam-app.png", use_container_width=True)
    st.title("🛡️ Spam Detection System")
    st.info("Detect spam messages with high accuracy using our Neural Network-powered engine.")
    
    st.subheader("💡 Tips")
    st.write("- Paste full message text for best results.")
    st.write("- Short messages may have lower confidence.")
    
    st.divider()
    st.write("Developed by **Au Amores**")

# --- MAIN UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📩 Message Analyzer")
    st.write("Paste your message below to check for potential security threats or spammy content.")
    
    user_input = st.text_area(
        "Enter your message:",
        placeholder="e.g. Last chance to claim your reward! Click bit.ly/spam-link now!",
        height=200,
        key='user_input_val'
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        if st.button("🚀 Analyze Now"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a message.")
            else:
                with st.spinner("Analyzing message patterns..."):
                    pred = predict_spam(user_input)
                    confidence = pred * 100
                    
                    st.session_state['pred'] = pred
                    st.session_state['confidence'] = confidence
                    st.session_state['input'] = user_input

with col2:
    st.write("### Analysis Dashboard")
    if 'pred' in st.session_state:
        pred = st.session_state['pred']
        confidence = st.session_state['confidence']
        
        if pred > 0.5:
            st.markdown(f"""
            <div class="result-card" style="background-color: rgba(245, 101, 101, 0.1);">
                <p class="spam-label">🚨 SPAM DETECTED</p>
                <p style="font-size: 18px;">Probability: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.image("assets/spam-app.png", width=150) # Using existing asset as visual feedback
            st.warning("This message exhibits strong characteristics of spam or phishing.")
        else:
            st.markdown(f"""
            <div class="result-card" style="background-color: rgba(72, 187, 120, 0.1);">
                <p class="ham-label">✅ LEGITIMATE MESSAGE</p>
                <p style="font-size: 18px;">Probability: <b>{100 - confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.image("assets/notspam.png", width=150) # Using existing asset
            st.success("This message appears safe and legitimate.")
            
        st.progress(int(confidence) if pred > 0.5 else int(100 - confidence))
    else:
        st.info("Analysis results will appear here after clicking 'Analyze Now'.")

# --- EXAMPLES SECTION ---
st.divider()
st.subheader("📝 Try an Example")
st.write("Click a button to test the detector with common message types:")
ex_col1, ex_col2, ex_col3 = st.columns(3)

examples = [
    ("Standard Message", "Hey, are we still meeting for lunch today? Let me know!"),
    ("Spam Win", "CONGRATULATIONS! You've been selected to win a $1000 Walmart Gift Card. Click here to claim!"),
    ("Phishing Link", "Your package is waiting for delivery. Please confirm your address at: http://bit.ly/malicious-link")
]

if ex_col1.button(f"🟢 {examples[0][0]}"):
    st.session_state['user_input_val'] = examples[0][1]
    st.rerun()

if ex_col2.button(f"🔴 {examples[1][0]}"):
    st.session_state['user_input_val'] = examples[1][1]
    st.rerun()

if ex_col3.button(f"🟠 {examples[2][0]}"):
    st.session_state['user_input_val'] = examples[2][1]
    st.rerun()




