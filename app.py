import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartCare | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Load model files (cached, with friendly error handling)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "heart_disease_model.pkl")
    selector = joblib.load(BASE_DIR / "feature_selector.pkl")
    return model, selector


try:
    model, selector = load_artifacts()
except Exception as e:
    st.error(
        "Model files could not be loaded. Make sure `heart_disease_model.pkl` and "
        "`feature_selector.pkl` are in the same folder as this app."
    )
    st.exception(e)
    st.stop()


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #F5F8FC; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #13315C 100%);
}
section[data-testid="stSidebar"] * { color: #E6EEF8 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
.brand {
    font-size: 26px; font-weight: 800; letter-spacing: .3px; margin-bottom: 0;
}
.brand-sub { font-size: 13px; opacity: .75; margin-top: -4px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0B2545 0%, #087F8C 100%);
    border-radius: 18px;
    padding: 40px 44px;
    color: white;
    box-shadow: 0 10px 30px rgba(11,37,69,.18);
    margin-bottom: 26px;
}
.hero h1 { color: white; font-size: 38px; font-weight: 800; margin: 0 0 8px 0; }
.hero p  { color: #D6E6F2; font-size: 17px; margin: 0; max-width: 720px; }

/* Page headers */
.page-title { font-size: 30px; font-weight: 800; color: #0B2545; margin-bottom: 2px; }
.page-sub   { font-size: 15px; color: #64748B; margin-bottom: 18px; }

/* Cards */
.card {
    background: white;
    border: 1px solid #E3EAF3;
    border-radius: 14px;
    padding: 22px 22px;
    height: 100%;
    box-shadow: 0 2px 8px rgba(15,23,42,.04);
}
.card .icon { font-size: 28px; }
.card h4 { color: #0B2545; margin: 8px 0 6px 0; font-size: 17px; font-weight: 700; }
.card p  { color: #64748B; margin: 0; font-size: 14px; line-height: 1.55; }

/* Step badges */
.step {
    display: flex; gap: 14px; align-items: flex-start;
    background: white; border: 1px solid #E3EAF3; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px;
}
.step-num {
    min-width: 32px; height: 32px; border-radius: 50%;
    background: #087F8C; color: white; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-text { color: #1E293B; font-size: 15px; padding-top: 4px; }

/* Section label inside form */
.section-label {
    color: #0B2545; font-size: 17px; font-weight: 700;
    border-left: 4px solid #087F8C; padding-left: 10px; margin: 6px 0 10px 0;
}

/* Result cards */
.result {
    border-radius: 16px; padding: 26px 28px; color: white; margin-top: 8px;
}
.result.pos { background: linear-gradient(135deg, #B42318, #E5484D); }
.result.neg { background: linear-gradient(135deg, #067647, #12B76A); }
.result h2 { color: white; margin: 0 0 6px 0; font-size: 26px; font-weight: 800; }
.result p  { color: rgba(255,255,255,.92); margin: 0; font-size: 15px; }

/* Buttons */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background-color: #087F8C; color: white; border-radius: 10px;
    height: 48px; width: 100%; font-weight: 700; font-size: 16px; border: none;
    transition: all .15s ease;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #05636D; color: white; transform: translateY(-1px);
}

/* Metric tiles */
div[data-testid="stMetric"] {
    background: white; border: 1px solid #E3EAF3; border-radius: 12px; padding: 14px 16px;
}

.disclaimer {
    background: #FFF8E6; border: 1px solid #F5D98B; color: #7A5B00;
    border-radius: 10px; padding: 12px 16px; font-size: 13.5px; margin-top: 16px;
}
.footer { text-align:center; color:#94A3B8; font-size:13px; margin-top: 30px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, text: str):
    st.markdown(
        f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def select_option(label, options: dict, default_key, help_text=None):
    """Selectbox that shows readable labels but returns the numeric code."""
    keys = list(options.keys())
    choice = st.selectbox(
        label,
        keys,
        index=keys.index(default_key),
        help=help_text,
    )
    return options[choice]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="brand">❤️ HeartCare</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Heart Disease Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🩺 Prediction", "📘 About Project", "⚙️ Model Information"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Machine Learning Based Heart Disease Prediction System")
    st.caption("For educational purposes only.")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Heart Disease Prediction System</h1>
            <p>A machine learning application that analyses patient health attributes
            and predicts the likelihood of heart disease in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card("🧾", "13 Clinical Inputs", "Age, blood pressure, cholesterol, ECG, heart rate and more.")
    with c2:
        card("🤖", "Machine Learning", "A trained classifier with feature selection for focused predictions.")
    with c3:
        card("⚡", "Instant Results", "Get the model's prediction and confidence with a single click.")

    st.markdown("<br>", unsafe_allow_html=True)
    page_header("How It Works")

    steps = [
        "Open the <b>Prediction</b> page from the sidebar.",
        "Enter the patient's information in the form.",
        "Click <b>Predict Heart Disease</b>.",
        "Review the prediction, probability and input summary.",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(
            f'<div class="step"><div class="step-num">{i}</div><div class="step-text">{s}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
elif page == "🩺 Prediction":
    page_header("Patient Prediction", "Fill in the patient details below and run the model.")

    SEX = {"Male": 1, "Female": 0}
    CP = {
        "0 - Typical Angina": 0,
        "1 - Atypical Angina": 1,
        "2 - Non-anginal Pain": 2,
        "3 - Asymptomatic": 3,
    }
    FBS = {"No (≤ 120 mg/dl)": 0, "Yes (> 120 mg/dl)": 1}
    RESTECG = {
        "0 - Normal": 0,
        "1 - ST-T Wave Abnormality": 1,
        "2 - Left Ventricular Hypertrophy": 2,
    }
    EXANG = {"No": 0, "Yes": 1}
    SLOPE = {"0 - Upsloping": 0, "1 - Flat": 1, "2 - Downsloping": 2}
    CA = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    THAL = {"0 - Unknown / Null": 0, "1 - Fixed Defect": 1, "2 - Normal": 2, "3 - Reversible Defect": 3}

    with st.form("prediction_form"):
        with st.container(border=True):
            section_label("👤 Personal Information")
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age (years)", 1, 120, 50)
            with c2:
                sex = select_option("Sex", SEX, "Male")

        with st.container(border=True):
            section_label("🩸 Medical Measurements")
            c1, c2 = st.columns(2)
            with c1:
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120, 1)
                chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, 1)
                fbs = select_option("Fasting Blood Sugar", FBS, "No (≤ 120 mg/dl)")
            with c2:
                cp = select_option("Chest Pain Type", CP, "0 - Typical Angina")
                restecg = select_option("Resting ECG", RESTECG, "0 - Normal")
                thalach = st.number_input("Maximum Heart Rate (bpm)", 50, 250, 150, 1)

        with st.container(border=True):
            section_label("🫀 Heart Test Results")
            c1, c2 = st.columns(2)
            with c1:
                exang = select_option("Exercise Induced Angina", EXANG, "No")
                oldpeak = st.number_input(
                    "Oldpeak (ST depression)", 0.0, 10.0, 0.0, 0.1,
                    help="ST depression induced by exercise relative to rest.",
                )
                slope = select_option("Slope of Peak Exercise ST", SLOPE, "0 - Upsloping")
            with c2:
                ca = select_option("Number of Major Vessels (0-4)", CA, "0")
                thal = select_option("Thalassemia (Thal)", THAL, "0 - Unknown / Null")

        submitted = st.form_submit_button("Predict Heart Disease")

    if submitted:
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        input_df = pd.DataFrame([input_data])

        try:
            selected_input = selector.transform(input_df)
            prediction = int(model.predict(selected_input)[0])
            probability = (
                float(model.predict_proba(selected_input)[0][1])
                if hasattr(model, "predict_proba")
                else None
            )
        except Exception as e:
            st.error("Prediction failed. Please check the inputs and model files.")
            st.exception(e)
            st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        page_header("Prediction Result")

        left, right = st.columns([3, 2])

        with left:
            if prediction == 1:
                st.markdown(
                    '<div class="result pos"><h2>⚠️ Heart Disease Detected</h2>'
                    "<p>The model predicts a presence of heart disease for the given inputs.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result neg"><h2>✅ No Heart Disease Detected</h2>'
                    "<p>The model predicts no heart disease for the given inputs.</p></div>",
                    unsafe_allow_html=True,
                )

        with right:
            if probability is not None:
                st.metric("Probability of Heart Disease", f"{probability * 100:.1f}%")
                st.progress(min(max(probability, 0.0), 1.0))
            else:
                st.metric("Prediction Class", "Positive" if prediction == 1 else "Negative")

        with st.expander("📋 View Input Summary"):
            summary = pd.DataFrame(
                {"Feature": list(input_data.keys()), "Value": list(input_data.values())}
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="disclaimer"><b>Disclaimer:</b> This is an educational machine learning '
            "prediction and not a medical diagnosis. Please consult a qualified doctor for any "
            "health concerns.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "📘 About Project":
    page_header("About The Project", "Overview, objective and technologies used.")

    with st.container(border=True):
        st.markdown("#### Overview")
        st.write(
            "Heart Disease Prediction is a Data Science project developed using Python and "
            "Machine Learning. It explores how machine learning algorithms can analyse "
            "health-related data and generate predictions."
        )

    with st.container(border=True):
        st.markdown("#### Project Objective")
        st.write(
            "To develop a machine learning application that predicts heart disease using "
            "selected health attributes."
        )

    st.markdown("#### Technologies Used")
    t1, t2, t3 = st.columns(3)
    with t1:
        card("🐍", "Python", "Core programming language of the project.")
    with t2:
        card("📊", "Pandas & NumPy", "Data handling and numerical computation.")
    with t3:
        card("🧠", "Scikit-learn", "Model training and feature selection.")
    st.markdown("<br>", unsafe_allow_html=True)
    t4, t5, t6 = st.columns(3)
    with t4:
        card("🌐", "Streamlit", "Interactive web interface for the app.")
    with t5:
        card("💾", "Joblib", "Saving and loading the trained model files.")
    with t6:
        card("🎓", "BS Information Technology", "Developed as a Data Science project.")


# ---------------------------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------------------------
elif page == "⚙️ Model Information":
    page_header("Model Information", "How the prediction pipeline is built.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("🔎", "1. Feature Selector", "Selects the important input features used by the trained model.")
    with c2:
        card("🤖", "2. Prediction Model", "Generates predictions using the selected input features.")
    with c3:
        card("🧾", "3. Input Features", "The application uses 13 patient-related attributes.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Input Features")
    features = pd.DataFrame(
        {
            "Feature": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                        "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            "Description": [
                "Age in years", "Sex (1 = male, 0 = female)", "Chest pain type",
                "Resting blood pressure", "Serum cholesterol", "Fasting blood sugar > 120 mg/dl",
                "Resting ECG result", "Maximum heart rate achieved",
                "Exercise induced angina", "ST depression induced by exercise",
                "Slope of peak exercise ST segment", "Number of major vessels",
                "Thalassemia result",
            ],
        }
    )
    st.dataframe(features, use_container_width=True, hide_index=True)

    st.info("The model and feature selector are loaded from saved Pickle (.pkl) files.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">Heart Disease Prediction System &nbsp;|&nbsp; '
    "BS Information Technology &nbsp;|&nbsp; Data Science Project</div>",
    unsafe_allow_html=True,
)