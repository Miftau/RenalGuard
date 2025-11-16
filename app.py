# app.py
"""
RenalGuard Flask app (full).
Features:
 - Kidney Disease Risk Prediction (Clinical & Lifestyle models)
 - Heuristic condition inference & suggestions for kidney health
 - AI chat endpoint (Groq preferred, OpenAI fallback) - Kidney-focused
 - Optional Supabase chat-history persistence
 - Model auto-download from REMOTE_MODEL_BASE (if configured)
 - Health endpoint for uptime monitoring
 - Supabase Auth (email/password, OAuth)
 - Subscription management integrated into dashboards
 - Access control based on subscription plans
Requirements (install in your venv):
pip install flask pandas numpy scikit-learn joblib python-dotenv flask-mail requests supabase py-sdk-openai groq flask-cors
Note: package names may vary slightly for groq or supabase clients; adjust to what you actually use:
 - supabase: "supabase" or "supabase-py" (depending on venv)
 - groq: "groq" (if you plan to use Groq)
 - openai: "openai" (fallback)
"""
# ============================================================
# Imports & App Initialization
# ============================================================
import os
import bcrypt
import uuid
import time
import json
import requests
import markdown
import jwt
from datetime import datetime, timezone, timedelta
import hashlib
from typing import List, Tuple, Optional
from groq import Groq
from supabase import create_client, Client
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, session
)
from flask_cors import CORS
from flask_mail import Mail, Message
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from functools import wraps # For login_required decorator
# Optional SDKs — imported inside try/except so app still runs without them
try:
    from supabase import create_client as create_supabase_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False
try:
    # Groq client; if not installed, fallback to openai
    import groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
# App initialization
load_dotenv()
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])
# Email config (optional)
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True") == "True",
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=(
        os.getenv("MAIL_SENDER_NAME", "RenalGuard Contact"),
        os.getenv("MAIL_USERNAME")
    ),
)
mail = Mail(app)
@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}

# ============================================================
# Custom Jinja2 Filters
# ============================================================
@app.template_filter('datetime_from_iso')
def datetime_from_iso_filter(date_string):
    """Convert an ISO format date string (YYYY-MM-DDTHH:MM:SS.ssssss) to a datetime object."""
    if not date_string:
        return None
    try:
        # Handle different potential formats, e.g., with/without microseconds, with/without 'Z'
        # The most common format from Supabase is YYYY-MM-DDTHH:MM:SS.ssssss+ZZ:ZZ
        # But the basic YYYY-MM-DDTHH:MM:SS also occurs
        # datetime.fromisoformat() is quite robust for standard ISO formats
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        # If parsing fails, return None or raise an error as appropriate
        print(f"Warning: Could not parse date string '{date_string}' using fromisoformat.")
        try:
            # Fallback: Try parsing without microseconds if the string contains them
            # This is less robust and assumes a specific format, better to rely on fromisoformat if possible
            return datetime.strptime(date_string.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            print(f"Warning: Could not parse date string '{date_string}' with fallback method either.")
            return None

@app.template_filter('now')
def now_filter():
    """Return the current datetime object."""
    return datetime.now()

# ============================================================
# Configuration & Constants
# ============================================================
# Paths, model filenames & optional remote base
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- RenalGuard Model Files ---
CLINICAL_MODEL_FILE = os.getenv("CLINICAL_MODEL_FILE", "kidney_model_clinical.pkl")
CLINICAL_ENCODERS_FILE = os.getenv("CLINICAL_ENCODERS_FILE", "preprocessing_info_cli.pkl") # For kidney clinical
CLINICAL_TEMPLATE_FILE = os.getenv("CLINICAL_TEMPLATE_FILE", "kidney_clinical_template.csv") # Placeholder, might not need exact template

LIFESTYLE_MODEL_FILE = os.getenv("LIFESTYLE_MODEL_FILE", "kidney_model_lifestyle.pkl")
LIFESTYLE_ENCODERS_FILE = os.getenv("LIFESTYLE_ENCODERS_FILE", "preprocessing_info_gen.pkl") # For kidney lifestyle
LIFESTYLE_TEMPLATE_FILE = os.getenv("LIFESTYLE_TEMPLATE_FILE", "kidney_lifestyle_template.csv") # Placeholder, might not need exact template

REMOTE_MODEL_BASE = os.getenv("REMOTE_MODEL_BASE", "").rstrip("/")

# Optional Supabase (chat history persistence, auth)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_AVAILABLE:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)
        supabase = None
else:
    if SUPABASE_URL or SUPABASE_KEY:
        print("⚠️ Supabase credentials provided but 'supabase' package not available.")
    else:
        print("ℹ️ Supabase not configured; chat history persistence disabled.")

# AI provider configuration (Groq preferred, OpenAI fallback)
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")

use_groq = bool(GROQ_API_KEY and GROQ_AVAILABLE)
use_openai = bool(OPENAI_API_KEY and OPENAI_AVAILABLE)

if use_groq:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq configured for chat.")
elif use_openai:
    openai.api_key = OPENAI_API_KEY
    print("✅ OpenAI configured for chat fallback.")
else:
    print("⚠️ No AI provider configured (GROQ or OpenAI). Chat endpoint will return an error unless keys installed.")

# Input columns (forms) - These should match the features used in your training script
# Example placeholders - replace with actual feature names from your X_gen.columns and X_cli.columns
# For example, if your lifestyle model used ['RIDAGEYR', 'RIAGENDR', 'BMXBMI', ...]
# and clinical used ['LBXSCR', 'LBXBUN', 'LBXGLU', ...]
BASE_COLUMNS_KIDNEY_LIFESTYLE = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "DMDMARTL",
    "BMXBMI", "BMXWAIST",
    "SMQ020", "SMQ040", "SMQ077",  # smoking
    "ALQ101", "ALQ130",  # alcohol
    "PADACTTV", "PAQ605",  # physical activity
    "DIQ010",  # diabetes self-report
    "BPQ020",  # hypertension self-report
    "MCQ160E"  # family history of kidney disease
    # Add more as needed from your training script
]
# Example placeholders - replace with actual feature names from your training script
BASE_COLUMNS_KIDNEY_CLINICAL = [
    "LBXSCR", "LBXBUN", "LBXEGFR",  # kidney function markers
    "LBXGLU", "LBXGH",  # glucose/HbA1c
    "LBDHDD", "LBDLDL", "LBXTCA",  # lipids
    "BPXSY1", "BPXDI1",  # systolic/diastolic BP
    "LBXPLTSI"  # platelets
    # Add more as needed from your training script
]

# ============================================================
# Virtual Meeting System (Jitsi) Helpers
# ============================================================
def generate_jitsi_url(appointment_id, appointment_time_str):
    """
    Generates a unique Jitsi meeting URL based on appointment details.
    Uses a deterministic hash to ensure the same appointment gets the same room.
    """
    # Create a unique identifier for the meeting room
    # Combining appointment ID and time should be sufficient
    unique_identifier = f"{appointment_id}_{appointment_time_str}"
    # Use SHA-256 hash to create a deterministic, unique room name
    room_hash = hashlib.sha256(unique_identifier.encode()).hexdigest()
    # Truncate the hash for readability (e.g., first 12 characters)
    room_name = room_hash[:12]
    # Use a public Jitsi server or your own self-hosted one
    jitsi_server = os.getenv("JITSI_SERVER_URL", "https://meet.jit.si") # e.g., "https://your-jitsi-instance.com"
    return f"{jitsi_server}/{room_name}"

# ============================================================
# Notification System (Email) Helpers
# ============================================================
def send_appointment_reminder_email(appointment_id, user_email, user_name, doctor_name, appointment_time_str):
    """
    Sends an email reminder for an appointment.
    """
    try:
        msg = Message(
            subject="[RenalGuard] Appointment Reminder",
            recipients=[user_email],
            body=f"""
            Dear {user_name},
            This is a friendly reminder that you have an appointment scheduled with Dr. {doctor_name} on {appointment_time_str}.
            Please ensure you are ready for the session.
            Best regards,
            The RenalGuard Team
            """
        )
        mail.send(msg)
        print(f"✅ Reminder email sent to {user_email} for appointment {appointment_id}")
        # Optionally, log this to the 'notifications' table
        if supabase:
            supabase.table("notifications").insert({
                "user_id": session.get("user_id"), # This might need adjustment if called outside a request context
                "appointment_id": appointment_id,
                "type": "appointment_reminder",
                "message": f"Reminder for appointment with Dr. {doctor_name} on {appointment_time_str}",
                "channel": "email",
                "status": "sent"
            }).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to send reminder email for appointment {appointment_id}: {e}")
        # Optionally, log failure to the 'notifications' table
        try:
            if supabase:
                supabase.table("notifications").insert({
                    "user_id": session.get("user_id"), # This might need adjustment if called outside a request context
                    "appointment_id": appointment_id,
                    "type": "appointment_reminder",
                    "message": f"Failed to send reminder for appointment with Dr. {doctor_name} on {appointment_time_str}",
                    "channel": "email",
                    "status": "failed",
                    "error_details": str(e) # Store error details
                }).execute()
        except Exception as log_e:
            print(f"❌ Failed to log notification failure: {log_e}")
        return False

# ============================================================
# Utility Functions
# ============================================================
def ensure_model_files(timeout=60):
    required = [
        CLINICAL_MODEL_FILE, CLINICAL_ENCODERS_FILE, # CLINICAL_TEMPLATE_FILE, # Template might not be needed
        LIFESTYLE_MODEL_FILE, LIFESTYLE_ENCODERS_FILE, # LIFESTYLE_TEMPLATE_FILE, # Template might not be needed
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if not missing:
        return
    if not REMOTE_MODEL_BASE:
        raise FileNotFoundError(f"Missing model/encoder files: {missing}. Set REMOTE_MODEL_BASE to auto-download them.")

    print("🛰️ Missing files detected:", missing)
    for fname in missing:
        url = f"{REMOTE_MODEL_BASE}/{fname}"
        try:
            print(f"Downloading {fname} from {url} ...")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(fname, "wb") as fh:
                fh.write(r.content)
            print("✅", fname)
        except Exception as e:
            print("❌ failed to download", fname, e)

# Load models & encoders
def load_models():
    print("🔄 Loading models & encoders...")
    clinical_model = joblib.load(CLINICAL_MODEL_FILE)
    clinical_encoders_info = joblib.load(CLINICAL_ENCODERS_FILE)
    # CLINICAL_FEATURE_COLUMNS = clinical_template_df.columns.tolist() # Not needed if using original column names from training

    lifestyle_model = joblib.load(LIFESTYLE_MODEL_FILE)
    lifestyle_encoders_info = joblib.load(LIFESTYLE_ENCODERS_FILE)
    # LIFESTYLE_FEATURE_COLUMNS = lifestyle_template_df.columns.tolist() # Not needed if using original column names from training

    print("✅ Models loaded")
    return (clinical_model, clinical_encoders_info,
            lifestyle_model, lifestyle_encoders_info)

(clinical_model, clinical_encoders_info,
 lifestyle_model, lifestyle_encoders_info) = load_models()

# Prediction helper (clinical & lifestyle) - Adapted for Kidney Models
def prepare_and_predict(df_raw: pd.DataFrame, model_type: str) -> pd.DataFrame:
    if model_type not in ("clinical", "lifestyle"):
        raise ValueError("Invalid model_type")

    df = df_raw.copy()

    # If headerless (pandas will provide integer column names), map positional columns
    # This part might need adjustment based on how you plan to pass data
    # For now, assume headers exist matching training features
    if all(isinstance(c, int) for c in df.columns):
        if model_type == "clinical":
            df = df.iloc[:, :len(BASE_COLUMNS_KIDNEY_CLINICAL)]
            df.columns = BASE_COLUMNS_KIDNEY_CLINICAL[:df.shape[1]]
        else:
            df = df.iloc[:, :len(BASE_COLUMNS_KIDNEY_LIFESTYLE)]
            df.columns = BASE_COLUMNS_KIDNEY_LIFESTYLE[:df.shape[1]]

    # --- RenalGuard Specific Preprocessing ---
    # Clinical pipeline: Apply encoders loaded from training
    if model_type == "clinical":
        encoders_info = clinical_encoders_info
        model = clinical_model
        feature_columns = BASE_COLUMNS_KIDNEY_CLINICAL # Use the list of expected columns from training
    else: # lifestyle
        encoders_info = lifestyle_encoders_info
        model = lifestyle_model
        feature_columns = BASE_COLUMNS_KIDNEY_LIFESTYLE # Use the list of expected columns from training

    # Get the encoders and column lists
    encoders = encoders_info['encoders']
    num_cols = encoders_info['numerical_columns']
    cat_cols = encoders_info['categorical_columns']

    # Prepare the input dataframe with only the required features
    X_input = df.reindex(columns=feature_columns, fill_value=0) # Fill missing features with 0 or handle differently

    # Apply preprocessing steps as done during training
    # 1. Handle numerical columns: Impute with median (as done in training script)
    X_num_imputed = pd.DataFrame(
        X_input[num_cols].fillna(X_input[num_cols].median()), # Use median of input row or overall median if needed
        columns=num_cols,
        index=X_input.index
    )

    # 2. Handle categorical columns: Impute with 'missing', then encode
    X_cat_encoded = X_input[cat_cols].copy()
    for col in cat_cols:
        le = encoders[col]
        # Impute unseen labels or missing values with 'missing' first
        X_cat_encoded[col] = X_cat_encoded[col].fillna('missing')
        # Transform using the fitted encoder
        # Handle unseen labels by mapping them to the index of 'missing' if 'missing' was a category during training
        # This is complex. A simpler fallback: map unseen to the first category.
        try:
            X_cat_encoded[col] = le.transform(X_cat_encoded[col])
        except ValueError as e:
            print(f"Warning: Unseen label in {col}: {e}. Mapping to first category index.")
            X_cat_encoded[col] = 0 # Map to first category index as fallback

    # Combine processed numerical and categorical features
    X_processed = pd.concat([X_num_imputed, X_cat_encoded], axis=1)

    # Ensure column order matches training
    X_final = X_processed[feature_columns]

    # Convert to numpy array for prediction
    X = X_final.values

    # --- Prediction ---
    df_out_base = df.copy() # Keep original input for output
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1] # Probability of positive class (disease)
    else:
        # fallback - sigmoid of decision_function if available
        try:
            df_dec = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-df_dec))
        except Exception:
            probs = model.predict(X).astype(float)  # last resort, predict_proba is preferred

    preds = model.predict(X)

    # Build output DataFrame
    out_df = df_out_base.copy()
    out_df["Prediction"] = preds # 0 or 1
    out_df["Prob_Pos"] = np.round(probs, 4) # Probability of having the condition
    out_df["Risk_Level"] = out_df["Prob_Pos"].apply(lambda p: "High" if p > 0.66 else ("Medium" if p > 0.33 else "Low"))
    return out_df

# Heuristic diagnostic rules (RenalGuard-specific)
def get_likely_kidney_condition(
    age, bmi, serum_creatinine, blood_urea_nitrogen, egfr,
    glucose, hba1c, systolic_bp, diastolic_bp,
    hdl_cholesterol, total_cholesterol, triglycerides,
    diabetes_status=None, hypertension_status=None,
    smoking_status=None, alcohol_status=None,
    family_history_kidney=None
) -> Tuple[str, List[str]]:
    """Return (likely_condition, suggestions) based on simple heuristic rules for kidney health."""
    likely_condition = "Generalized Kidney Risk"
    suggestions: List[str] = []

    # normalize
    def to_float(v, default=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    age_v = to_float(age, 0)
    bmi_v = to_float(bmi, 0)
    scr_v = to_float(serum_creatinine, 0)
    bun_v = to_float(blood_urea_nitrogen, 0)
    egfr_v = to_float(egfr, 120) # eGFR typically starts high, lower is worse
    glu_v = to_float(glucose, 0)
    hba1c_v = to_float(hba1c, 0)
    sbp_v = to_float(systolic_bp, 0)
    dbp_v = to_float(diastolic_bp, 0)
    hdl_v = to_float(hdl_cholesterol, 0)
    tchol_v = to_float(total_cholesterol, 0)
    trigs_v = to_float(triglycerides, 0)

    # normalize some categorical inputs to lowercase strings for matching
    diabetes_s = str(diabetes_status).strip().lower() if diabetes_status is not None else ""
    hypertension_s = str(hypertension_status).strip().lower() if hypertension_status is not None else ""
    smoking_s = str(smoking_status).strip().lower() if smoking_status is not None else ""
    alcohol_s = str(alcohol_status).strip().lower() if alcohol_status is not None else ""
    family_hist_s = str(family_history_kidney).strip().lower() if family_history_kidney is not None else ""

    # Rule set (ordered — earlier matches stronger)
    # Criteria for CKD often involve eGFR < 60 mL/min/1.73m2 and/or albuminuria (not available here, but ACR could be added if URXU_L is available)
    # Using eGFR and Creatinine/BUN as proxies
    if egfr_v < 60 or scr_v > 1.2 or bun_v > 20: # Rough thresholds for concern
        likely_condition = "Chronic Kidney Disease (CKD) - Potential"
        suggestions = [
            "Consult a nephrologist immediately for further evaluation.",
            "Get a urine test for albumin (ACR) if not already done.",
            "Strictly manage blood pressure and blood sugar if applicable."
        ]
    elif (sbp_v >= 140 or dbp_v >= 90) and egfr_v < 90:
        likely_condition = "Hypertensive Nephropathy Risk"
        suggestions = [
            "Optimize antihypertensive therapy to target <130/80 mmHg.",
            "Monitor kidney function (eGFR, Creatinine) regularly.",
            "Reduce sodium intake (<2g/day)."
        ]
    elif diabetes_s in {"1", "yes", "true", "y"} and egfr_v < 90:
        likely_condition = "Diabetic Nephropathy Risk"
        suggestions = [
            "Tight glycemic control (HbA1c < 7% or as advised).",
            "Consider ACE inhibitors or ARBs if not contraindicated.",
            "Annual screening for microalbuminuria."
        ]
    elif hba1c_v > 7.0 or glu_v > 140:
        likely_condition = "Diabetes-related Kidney Risk"
        suggestions = [
            "Consult an endocrinologist for blood sugar management.",
            "Discuss medications like SGLT2 inhibitors which may protect kidneys.",
            "Follow a diabetic diet."
        ]
    elif smoking_s in {"1", "yes", "true", "y"} and (egfr_v < 90 or scr_v > 1.0):
        likely_condition = "Smoking-related Kidney Risk"
        suggestions = [
            "Immediate smoking cessation is crucial.",
            "Smoking accelerates kidney function decline.",
            "Consider nicotine replacement therapy or counseling."
        ]
    elif age_v > 60 and egfr_v < 90:
        likely_condition = "Age-related Kidney Function Decline"
        suggestions = [
            "Regular monitoring of eGFR and Creatinine is recommended.",
            "Maintain adequate hydration.",
            "Review medications for potential nephrotoxicity."
        ]
    elif family_hist_s in {"1", "yes", "true", "y"} and egfr_v < 90:
        likely_condition = "Hereditary Kidney Disease Risk"
        suggestions = [
            "Discuss family history with a doctor.",
            "Consider genetic testing if polycystic kidney disease is suspected.",
            "More frequent kidney function checks may be needed."
        ]
    elif (bmi_v > 35 and (sbp_v > 130 or dbp_v > 85)) or (trigs_v > 200 and hdl_v < 40):
        likely_condition = "Metabolic Syndrome-related Kidney Risk"
        suggestions = [
            "Focus on weight loss through diet and exercise.",
            "Address lipid abnormalities with diet and potentially medication.",
            "Manage blood pressure aggressively."
        ]
    elif (scr_v < 1.0 and bun_v < 20 and egfr_v >= 90 and sbp_v < 120 and dbp_v < 80):
        likely_condition = "Low Kidney Risk"
        suggestions = [
            "Maintain a healthy lifestyle.",
            "Continue regular checkups."
        ]
    else:
        # Default if no strong rule matches
        likely_condition = "Generalized Kidney Risk"
        suggestions = [
            "Discuss results with your primary care doctor or a nephrologist.",
            "Consider targeted tests (eGFR, Creatinine, Urinalysis) if concerned."
        ]

    return likely_condition, suggestions


# Chat helpers (Groq / OpenAI) + Supabase persistence
def save_chat_message(user_id: str, role: str, message: str):
    if supabase:
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": role,
                "message": message
            }).execute()
        except Exception as e:
            # don't fail the whole request because of DB issues
            print("⚠️ Supabase insert failed:", e)

def call_groq_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    # Minimal Groq chat usage - adjust to your groq SDK version
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        raise RuntimeError("Groq not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-13b"),  # choose desired model
            messages=messages,
            temperature=float(os.getenv("GROQ_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
        )
        # the exact response structure may vary — adapt to your groq client
        text = response.choices[0].message["content"]
        return text
    except Exception as e:
        print("Groq chat error:", e)
        raise

def call_openai_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    if not OPENAI_API_KEY or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 512)),
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("OpenAI chat error:", e)
        raise

# Authentication & Authorization Helpers
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id") and not session.get("doctor_id") and not session.get("admin_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def get_user_role(user_id):
    """Helper to fetch user role from Supabase.
    Checks the 'users', 'doctors', and 'admins' tables based on the Supabase Auth ID.
    Falls back to 'user' if the user is not found in any table.
    """
    try:
        # Check 'users' table first
        user_data = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if user_data.data:
            return user_data.data.get("role", "user")
    except Exception as e:
        # If not found in 'users', proceed to check other tables
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'users' table for {user_id}: {e}")

    try:
        # Check 'doctors' table
        doctor_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
        if doctor_data.data:
            # If found in doctors table, their role is 'doctor'
            return "doctor"
    except Exception as e:
        # If not found in 'doctors', proceed to check 'admins'
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'doctors' table for {user_id}: {e}")

    try:
        # Check 'admins' table
        # The 'admins' table likely has 'user_id' column matching the Supabase Auth ID
        admin_data = supabase.table("admins").select("id").eq("id", user_id).single().execute()
        if admin_data.data:
            # If found in admins table, their role is 'admin'
            return "admin"
    except Exception as e:
        # If not found in 'admins' either, log error (if it's not the expected '0 rows')
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'admins' table for {user_id}: {e}")

    # If not found in any table, default to 'user' or handle as needed
    print(f"User {user_id} not found in any role table ('users', 'doctors', 'admins'). Assigning default role 'user'.")
    return "user"

def get_user_subscription_status(user_id):
    """
    Helper to fetch user's current subscription status from Supabase.
    Admin users are automatically considered unrestricted (active and paid).
    """
    try:
        # Step 1: Check if user is admin first
        role = get_user_role(user_id)
        if role == "admin":
            # Admins should have full access even without a subscription
            return "active", False

        # Step 2: For normal users, fetch actual subscription
        subs_data = (
            supabase.table("user_subscriptions")
            .select("status, subscription_plans(name, is_free)")
            .eq("user_id", user_id)
            .neq("status", "cancelled")
            .order("start_date", desc=True)
            .execute()
        )
        if subs_data.data:
            latest_sub = subs_data.data[0]
            plan_info = latest_sub.get("subscription_plans", {})
            is_free_plan = plan_info.get("is_free", True)
            return latest_sub["status"], is_free_plan
        else:
            # No active subscriptions found, default to free
            return "inactive", True
    except Exception as e:
        print(f"Error fetching subscription for user {user_id}: {e}")
        # Default to safest option (free/restricted)
        return "inactive", True

def handle_supabase_auth_session(session_data):
    """
    Process the session data returned from Supabase auth and set Flask session.
    Determines the user's role by checking the 'users', 'doctors', and 'admins' tables.
    """
    user_info = jwt.decode(session_data["access_token"], options={"verify_signature": False}, algorithms=["RS256"], audience="authenticated")
    user_id = user_info["sub"]
    email = user_info.get("email")
    user_name = user_info.get("user_metadata", {}).get("name") or email

    # Determine role by checking tables in a specific order
    # You might want to adjust the logic here depending on how you assign roles initially
    # e.g., via admin panel, registration form, or by checking the existence in specific tables.
    role = "user" # Default role

    # Check 'doctors' table first (or the order you prefer)
    try:
        doctor_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
        if doctor_data.data:
            role = "doctor"
    except Exception as e:
        # If not found in doctors, continue checking
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'doctors' table for {user_id}: {e}")

    # Check 'admins' table if role is still default
    if role == "user": # Only check admins if not already determined to be a doctor
        try:
            admin_data = supabase.table("admins").select("id").eq("id", user_id).single().execute()
            if admin_data.data:
                role = "admin"
        except Exception as e:
            # If not found in admins, continue checking users or keep default
            if "PGRST116" not in str(e) or "0 rows" not in str(e):
                print(f"Error checking 'admins' table for {user_id}: {e}")

    # Check 'users' table if role is still default
    # This also ensures the user profile exists in the 'users' table if needed for other data
    if role == "user":
        user_meta = supabase.table("users").select("*").eq("id", user_id).execute().data
        if not user_meta:
            # First-time user (or user not in 'users' table but defaulting to 'user' role): create profile
            # This assumes the user should exist in the 'users' table regardless of their primary role for general data.
            try:
                supabase.table("users").insert({
                    "id": user_id, # Using Supabase Auth ID as primary key
                    "email": email,
                    "name": user_name,
                    "role": "user" # The role here might be less critical if determined above, but set for consistency
                }).execute()
                # Fetch the data back to ensure we have it
                user_meta = [{"id": user_id, "role": "user", "name": user_name, "email": email}]
            except Exception as e:
                print(f"Error creating user profile in 'users' table for {user_id}: {e}")
                # If creation fails, we might not have user details, but we have the role determined above
                user_meta = [{"id": user_id, "role": role, "name": user_name, "email": email}]
        else:
            # User exists in 'users' table, potentially update name/email if changed in auth
            # Or just use the data fetched
            pass

    # Set Flask session variables
    session.clear()
    session["user_id"] = user_id
    session["role"] = role # Use the role determined from the tables
    session["user_name"] = user_name
    # Optionally, store email if needed elsewhere
    session["user_email"] = email

# Access Control Decorators
def check_subscription_access(f):
    """
    Decorator to check if the user has access to restricted features based on their subscription.
    Applies to features like /form, chat, booking.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")
        # If not logged in, treat as free user (restricted)
        if not user_id:
            # For non-logged-in users, check session-based limit for /form
            if request.endpoint == 'predict': # Only apply to the form submission route
                last_form_time = session.get('last_form_time')
                if last_form_time:
                    last_time = datetime.fromisoformat(last_form_time)
                    now = datetime.now()
                    if now - last_time < timedelta(days=30): # 30 days for example
                        flash("You can only submit the form once per month as a non-logged-in user.", "warning")
                        return redirect(url_for('form'))
            # Redirect to login or show restricted message for other features
            elif request.endpoint in ['chat', 'book_appointment']:
                flash("Please log in to access this feature.", "warning")
                return redirect(url_for('login'))
            # Allow access to the decorated function
            return f(*args, **kwargs)

        # User is logged in, fetch subscription status
        try:
            sub_status, is_free_plan = get_user_subscription_status(user_id)
        except Exception:
            # If there's an error fetching subscription, default to restricted
            flash("Error checking subscription status. Please try again later.", "danger")
            return redirect(url_for('user_dashboard'))

        if not is_free_plan and sub_status == "active":
            # Paid subscriber, allow access
            return f(*args, **kwargs)

        # Free subscriber or no active subscription
        if request.endpoint == 'predict':
            # Check usage limit for /form
            try:
                # Get the first record of the current month for this user
                start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                records = supabase.table("records").select("id").eq("user_id", user_id).gte("created_at", start_of_month.isoformat()).execute().data
                if len(records) >= 1: # Change 1 to your desired limit for paid users if different
                    flash("You have reached your monthly limit for form submissions.", "warning")
                    return redirect(url_for('user_dashboard'))
            except Exception as e:
                print(f"Error checking form limit for user {user_id}: {e}")
                flash("Error checking usage limit. Please try again.", "danger")
                return redirect(url_for('user_dashboard'))
        elif request.endpoint in ['chat', 'book_appointment']:
            # Restrict chat and booking for free users
            flash("This feature is available for paid subscribers only.", "warning")
            return redirect(url_for('user_dashboard'))

        # Allow access to the decorated function (e.g., for /form if limit not reached)
        return f(*args, **kwargs)
    return decorated_function

# ensure files on startup
ensure_model_files()

# ============================================================
# Routes - Authentication
# ============================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "user").strip().lower()  # expected: 'user', 'doctor' (or admin via admin UI)
        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("register"))

        # Check if user exists
        try:
            existing = supabase.table("users").select("id, email").eq("email", email).execute()
            if existing and existing.data:
                flash("Email already registered. Try logging in.", "warning")
                return redirect(url_for("login"))
        except Exception as e:
            print("Supabase check error:", e)
            flash("Registration currently unavailable. Try again later.", "danger")
            return redirect(url_for("register"))

        # Hash password with bcrypt
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Create user in users table (role included)
        try:
            res = supabase.table("users").insert({
                "name": name,
                "email": email,
                "password_hash": hashed_pw,
                "role": role
            }).execute()
            # extract created user id if available
            new_user = res.data[0] if res and res.data else None
        except Exception as e:
            print("Supabase insert user error:", e)
            flash("Failed to create account. Try again later.", "danger")
            return redirect(url_for("register"))

        # If doctor, create doctor profile row
        try:
            if role == "doctor":
                user_id = new_user.get("id") if new_user else None
                if user_id:
                    supabase.table("doctors").insert({
                        "user_id": user_id,
                        "specialization": request.form.get("specialization", "General Nephrology"),
                        "bio": request.form.get("bio", ""),
                        "consultation_fee": float(request.form.get("consultation_fee", 0))
                    }).execute()
        except Exception as e:
            print("Warning: failed to create doctor profile:", e)

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))
    # GET
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Please provide both email and password.", "warning")
            return redirect(url_for("login"))

        # ---- STEP 1: Check Users Table ----
        try:
            user_resp = supabase.table("users").select("*").eq("email", email).limit(1).execute()
        except Exception as e:
            print("Supabase user query error:", e)
            flash("Login temporarily unavailable. Please try again later.", "danger")
            return redirect(url_for("login"))

        if user_resp and user_resp.data:
            user = user_resp.data[0]
            stored_hash = user.get("password_hash")
            if stored_hash:
                try:
                    if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                        session.clear()
                        session["user_id"] = user.get("id")
                        session["user_name"] = user.get("name")
                        # --- STEP 1A: Detect Role ---
                        # Default role is 'user', unless doctor or admin
                        role = user.get("role", "user")
                        # If not admin, check if this user is a doctor
                        if role != "admin":
                            doctor_check = supabase.table("doctors").select("id").eq("id", user["id"]).execute()
                            if doctor_check.data:
                                role = "doctor"
                                session["doctor_id"] = doctor_check.data[0]["id"]
                        session["role"] = role
                        # --- STEP 1B: Redirect Based on Role ---
                        if role == "admin":
                            flash("Welcome back, Admin!", "success")
                            return redirect(url_for("admin_dashboard"))
                        elif role == "doctor":
                            flash("Welcome Doctor!", "success")
                            return redirect(url_for("doctor_dashboard"))
                        else:
                            flash("Welcome back!", "success")
                            return redirect(url_for("user_dashboard"))
                except ValueError as e:
                    print("Password hash error:", e)
                    flash("Error verifying credentials. Contact support.", "danger")
                    return redirect(url_for("login"))

        # ---- STEP 2: Check Admins Table (Fallback) ----
        try:
            admin_resp = supabase.table("admins").select("*").eq("email", email).limit(1).execute()
            if admin_resp and admin_resp.data:
                admin = admin_resp.data[0]
                stored_hash = admin.get("password_hash")
                if stored_hash and bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                    session.clear()
                    session["user_id"] = admin.get("id")
                    session["role"] = "admin"
                    session["user_name"] = admin.get("name")
                    flash("Welcome back, Admin!", "success")
                    return redirect(url_for("admin_dashboard"))
        except Exception as e:
            print("Supabase admin lookup error:", e)

        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    supabase.auth.sign_out() # Sign out from Supabase
    session.pop("user_id", None)
    session.pop("role", None)
    session.pop("user_name", None)
    session.pop("user_email", None) # Clear email
    flash("Logged out successfully.")
    return redirect(url_for("login"))

# --- Social Login Routes ---
@app.route("/auth/<provider>")
def social_login(provider):
    providers = {"google", "facebook", "github"}
    if provider not in providers:
        flash("Unsupported provider", "danger")
        return redirect(url_for("login"))
    redirect_url = url_for("auth_callback", provider=provider, _external=True)
    # Supabase OAuth URL
    auth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider={provider}&redirect_to={redirect_url}"
    return redirect(auth_url)

@app.route("/auth/callback/<provider>")
def auth_callback(provider):
    code = request.args.get("code")
    if not code:
        flash("Authentication failed", "danger")
        return redirect(url_for("login"))

    # Exchange code for session
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=authorization_code",
        json={"code": code, "redirect_uri": url_for("auth_callback", provider=provider, _external=True)},
        headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    )
    data = res.json()
    if "access_token" not in data:
        flash("Login failed", "danger")
        return redirect(url_for("login"))

    handle_supabase_auth_session(data) # Process session and set Flask session
    flash("Signed in successfully!", "success")
    role = session["role"]
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    elif role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    else:
        return redirect(url_for("user_dashboard"))

# ============================================================
# Routes - Main UI Pages (Adjusted for RenalGuard)
# ============================================================

@app.route("/")
def index():
    # Option 1: Render the main form directly on the index page
    return render_template(
        "form.html", # Use the form template as the main page
        BASE_COLUMNS_CLINICAL=BASE_COLUMNS_KIDNEY_CLINICAL,
        BASE_COLUMNS_LIFESTYLE=BASE_COLUMNS_KIDNEY_LIFESTYLE
    )
    # Option 2: Render a simple landing page that redirects to /form
    # return render_template("index.html") # If you create a simple index.html

@app.route("/form")
def form():
    # Check access for non-logged-in users only
    if not session.get("user_id"):
        if last_form_time := session.get('last_form_time'):
            last_time = datetime.fromisoformat(last_form_time)
            now = datetime.now()
            if now - last_time < timedelta(days=30): # 30 days
                flash("You can only access the form once per month as a non-logged-in user.", "warning")
                return redirect(url_for('index'))
    return render_template(
        "form.html",
        BASE_COLUMNS_CLINICAL=BASE_COLUMNS_KIDNEY_CLINICAL,
        BASE_COLUMNS_LIFESTYLE=BASE_COLUMNS_KIDNEY_LIFESTYLE
    )

# ============================================================
# Routes - Prediction & AI (RenalGuard Specific)
# ============================================================


@app.route("/predict", methods=["POST"])
@check_subscription_access # Apply access control
def predict():
    try:
        # --- Get Model Type ---
        raw_type = (request.form.get("model_type") or "clinical").lower()
        # Map user-friendly names to internal types
        model_map = {
            "kidney_clinical": "clinical",
            "clinical": "clinical",
            "kidney_lifestyle": "lifestyle",
            "lifestyle": "lifestyle"
        }
        model_type = model_map.get(raw_type, "clinical") # Default to clinical if unknown

        # --- Get Data ---
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            df = pd.read_csv(uploaded_file)
        else:
            # Get the base columns for the selected model type
            if model_type == "clinical":
                base_cols = BASE_COLUMNS_KIDNEY_CLINICAL # Use the updated list from app.py
            else: # lifestyle
                base_cols = BASE_COLUMNS_KIDNEY_LIFESTYLE # Use the updated list from app.py

            user_data = {}
            for c in base_cols:
                # Get the value directly using the feature name as the form input name
                val = request.form.get(c)
                # Optional: Add logic here if certain values need to be converted
                # (e.g., 'Yes'/'No' to 1/0, or string numbers to floats)
                # For now, assume the form sends the correct data type or a string representation
                user_data[c] = val
            df = pd.DataFrame([user_data])

        # --- Predict ---
        # Ensure the model type passed here matches the one used to select base_cols
        results = prepare_and_predict(df, model_type)

        # --- Post-Success Logic (saving records, session update) ---
        if user_id := session.get("user_id"):
            try:
                supabase.table("records").insert({
                    "user_id": user_id,
                    "consultation_type": f"kidney_{model_type}", # Store kidney-specific type
                    "health_score": float(results.iloc[0]["Prob_Pos"]),
                    "recommendation": "See results page for details."
                }).execute()
            except Exception as e:
                print(f"Error saving record for user {user_id}: {e}")
                flash("⚠️ Warning: Result not saved to your history.", "warning")
        else:
            session['last_form_time'] = datetime.now().isoformat()

        # --- Prepare Results for Template ---
        single = results.iloc[0]
        prob = float(single["Prob_Pos"])
        risk = single["Risk_Level"]
        readable = (
            "Kidney Disease Risk Detected" if single["Prediction"] == 1
            else "No Kidney Disease Risk Detected"
        )

        # --- Heuristic Inference ---
        # Map result columns back to function arguments for get_likely_kidney_condition
        # Use .get() with defaults to handle missing columns gracefully
        likely_condition, suggestions = get_likely_kidney_condition(
            age=single.get("RIDAGEYR"),
            bmi=single.get("BMXBMI", 0),
            serum_creatinine=single.get("LBXSCR", 0),
            blood_urea_nitrogen=single.get("LBXBUN", 0),
            egfr=single.get("LBXEGFR", 120), # Default high eGFR
            glucose=single.get("LBXGLU", 0),
            hba1c=single.get("LBXGH", 0),
            systolic_bp=single.get("BPXSY1", 0),
            diastolic_bp=single.get("BPXDI1", 0),
            hdl_cholesterol=single.get("LBDHDD", 0),
            total_cholesterol=single.get("LBXTCA", 0),
            triglycerides=single.get("LBXTR", 0),
            diabetes_status=single.get("DIQ010"), # Assuming DIQ010 is diabetes flag
            hypertension_status=single.get("BPQ020"), # Assuming BPQ020 is hypertension flag
            smoking_status=single.get("SMQ020"), # Assuming SMQ020 is smoking flag
            alcohol_status=single.get("ALQ101"), # Assuming ALQ101 is alcohol flag
            family_history_kidney=single.get("MCQ160E") # Assuming MCQ160E is family history flag
        )

        # --- Save Results CSV ---
        fname = f"kidney_{model_type}_pred_{uuid.uuid4().hex[:8]}.csv"
        save_path = os.path.join(RESULTS_DIR, fname)
        results.to_csv(save_path, index=False)
        download_link = url_for("static", filename=f"results/{fname}")

        # --- Render Result Template ---
        return render_template(
            "result.html",
            result=readable,
            prob=prob,
            risk=risk,
            likely_condition=likely_condition,
            suggestions=suggestions,
            tables=[results.to_html(classes="table table-striped", index=False)],
            download_link=download_link,
            model_type=model_type
        )

    except Exception as e:
        print("Prediction error:", e)
        import traceback
        traceback.print_exc() # Print the full traceback for detailed debugging
        flash(f"Error processing prediction: {str(e)}", "danger")
        return redirect(url_for("form"))



# AI chat endpoint (JSON API) - RenalGuard specific
@app.route("/consult", methods=["POST"])
def consult():
    """
    JSON input:
    { "user_id": "user-123", "message": "I have high creatinine, what does it mean?" }
    Response:
    { "reply": "...", "saved": true/false }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' in JSON body"}), 400

    user_msg = data["message"]
    user_id = data.get("user_id", f"anon-{uuid.uuid4().hex[:8]}")

    # --- RenalGuard Specific System Prompt ---
    system_prompt = os.getenv("CHAT_SYSTEM_PROMPT", "You are RenalConsult, a medically informed assistant specializing in kidney health. Provide safe, conservative guidance related to kidney function, risk factors, and general wellness. Always advise seeing a nephrologist for definitive diagnosis and treatment.")

    # store user message (best-effort)
    try:
        save_chat_message(user_id, "user", user_msg)
    except Exception as e:
        print("Warning: save chat failed:", e)

    # call AI provider
    try:
        if use_groq:
            ai_reply = call_groq_chat(user_msg, system_prompt=system_prompt)
        elif use_openai:
            ai_reply = call_openai_chat(user_msg, system_prompt=system_prompt)
        else:
            return jsonify({"error": "No AI provider configured (set GROQ_API_KEY or OPENAI_API_KEY)."}), 500
    except Exception as e:
        print("AI call failed:", e)
        return jsonify({"error": "AI provider error", "details": str(e)}), 500

    # persist assistant reply
    try:
        save_chat_message(user_id, "assistant", ai_reply)
    except Exception as e:
        print("Warning: save chat failed:", e)

    return jsonify({"reply": ai_reply, "saved": bool(supabase)}), 200

@app.route("/chat", methods=["GET", "POST"])
@limiter.limit("5 per minute")
#@check_subscription_access # Apply access control
def chat():
    if request.method == "GET":
        chat_log = session.get("chat_log", [])
        return render_template("chat.html", chat_log=chat_log)
    else:
        data = request.get_json()
        user_message = data.get("message", "")
        chat_log = session.get("chat_log", [])
        chat_log.append({"role": "user", "message": user_message})

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": (
                        "You are RenalConsult, a compassionate AI nephrologist providing preventive kidney health advice. "
                        "Do not give medical diagnoses; only provide general wellness guidance based on kidney science."
                    )},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens = 1000
            )
            reply = response.choices[0].message.content.strip()
            formatted_reply = markdown.markdown(reply, extensions=["fenced_code", "nl2br"])
        except Exception as e:
            print("Groq connection error:", e)
            reply = "⚠️ Sorry, I'm having trouble connecting to my kidney consultation engine."

        chat_log.append({"role": "assistant", "message": formatted_reply})
        session["chat_log"] = chat_log[-10:]

        try:
            supabase.table("chat_logs").insert({
                "user_id": session.get("user_id", "guest"),
                "user_message": user_message,
                "bot_reply": formatted_reply,
            }).execute()
        except Exception as e:
            print("Logging error:", e)

        return jsonify({"reply": formatted_reply})

@app.route("/api/chat", methods=["POST"])
@limiter.limit("5 per minute")  # tighter limit for chat API
def ai_chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Check if user is logged in and has access (for API calls, check session or token)
    user_id = request.json.get("id") # Assuming API sends user_id
    if user_id:
        # Fetch subscription for API user
        try:
            sub_status, is_free_plan = get_user_subscription_status(user_id)
            if is_free_plan or sub_status != "active":
                return jsonify({"error": "AI chat access denied. Upgrade your subscription."}), 403
        except Exception:
            return jsonify({"error": "Subscription check failed."}), 500
    else:
        # For anonymous API calls, deny access
        return jsonify({"error": "AI chat requires authentication and a paid subscription."}), 403

    # Retrieve session chat (if applicable for API, maybe use DB or token-based session)
    # For simplicity here, we'll proceed without session for the API endpoint
    # In a real app, you'd manage state differently for APIs.

    # Prepare Groq request
    try:
        # --- RenalGuard Specific System Prompt for API ---
        system_prompt = "You are RenalConsult, an AI kidney health consultant. Provide informative, safe responses related to kidney function, risk factors, and preventive care. Always encourage users to see a nephrologist if symptoms persist or if results are concerning."
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",  # free + hosted
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7
            },
            timeout=15
        )
        data = response.json()
        ai_reply = data["choices"][0]["message"]["content"]

        # Basic logging (could log to DB instead)
        print(f"[CHAT LOG] User: {user_input}\nAI: {ai_reply}\n---")
        return jsonify({"reply": ai_reply})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "RenalConsult AI is currently unavailable. Please try again later."}), 500

# -----------------------------
# Utility: Clear chat
# -----------------------------
@app.route("/chat/clear")
def clear_chat():
    session.pop("chat_history", None)
    flash("Chat history cleared.")
    return redirect(url_for("chat"))

@app.route("/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    if not supabase:
        return jsonify({"error": "Chat history persistence is not configured."}), 400
    try:
        res = supabase.table("chat_history").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return jsonify({"history": res.data}), 200
    except Exception as e:
        print("Supabase fetch failed:", e)
        return jsonify({"error": "Failed to fetch history", "details": str(e)}), 500

# ============================================================
# Routes - Doctor Features (RenalGuard Specific)
# ============================================================

@app.route("/doctor/availability", methods=["GET", "POST"])
def doctor_availability():
    if session.get("role") != "doctor":
        flash("Doctor access only.", "warning")
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    # find doctor record
    doctor_data = supabase.table("doctors").select("id").eq("user_id", user_id).single().execute()
    if not doctor_data.data:
        flash("Doctor profile not found.", "danger")
        return redirect(url_for("login"))

    doctor_id = doctor_data.data["id"]

    if request.method == "POST":
        date_str = request.form.get("available_date")
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        try:
            available_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            supabase.table("doctor_availability").insert({
                "doctor_id": doctor_id,
                "available_date": str(available_date),
                "start_time": start_time,
                "end_time": end_time
            }).execute()
            flash("Availability added successfully.", "success")
        except Exception as e:
            print("Error adding availability:", e)
            flash("Failed to add availability.", "danger")
        return redirect(url_for("doctor_availability"))

    # GET request
    slots = supabase.table("doctor_availability").select("*").eq("doctor_id", doctor_id).order("available_date").execute().data
    return render_template("doctor_availability.html", slots=slots)

@app.route("/book", methods=["GET", "POST"])
@login_required
@check_subscription_access # Apply access control
def book_appointment():
    if session.get("role") != "user":
        flash("Login as a user to book an appointment.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        slot_id = request.form.get("slot_id")
        # Fetch slot
        slot_data = supabase.table("doctor_availability").select("*").eq("id", slot_id).single().execute()
        if not slot_data:
            flash("Invalid slot selected.", "danger")
            return redirect(url_for("book_appointment"))

        slot = slot_data.data
        if slot.get("is_booked"):
            flash("This slot has already been booked.", "danger")
            return redirect(url_for("book_appointment"))

        # Create appointment
        doctor_id = slot["doctor_id"]
        user_id = session.get("user_id")
        dt_str = f"{slot['available_date']} {slot['start_time']}"
        appointment_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

        # --- NEW: Generate Jitsi URL ---
        jitsi_url = generate_jitsi_url(slot_id, dt_str) # Use slot_id and time_str for uniqueness
        supabase.table("appointments").insert({
            "user_id": user_id,
            "doctor_id": doctor_id,
            "appointment_time": appointment_time.isoformat(),
            "status": "pending",
            "meeting_url": jitsi_url # Add the generated URL
        }).execute()

        # Mark slot as booked
        supabase.table("doctor_availability").update({"is_booked": True}).eq("id", slot_id).execute()

        # --- NEW: Send Appointment Reminder Email ---
        # Fetch doctor details for the email
        doctor_details = supabase.table("doctors").select("id, specialization").eq("id", doctor_id).single().execute().data
        user_details = supabase.table("users").select("name, email").eq("id", user_id).single().execute().data

        doctor_name = doctor_details.get("specialization", "Unknown") # Or fetch from user table if needed
        user_name = user_details.get("name")
        user_email = user_details.get("email")
        appointment_time_str = appointment_time.strftime("%Y-%m-%d at %H:%M")

        send_appointment_reminder_email(slot_id, user_email, user_name, doctor_name, appointment_time_str)

        flash("Appointment booked successfully!", "success")
        return redirect(url_for("user_dashboard"))

    # GET: get all available slots
    available_slots = supabase.table("doctor_availability").select(
        "id, doctor_id, available_date, start_time, end_time"
    ).eq("is_booked", False).execute().data

    doctors_data = supabase.table("doctors").select("id, specialization, bio").execute().data
    doctor_lookup = {d["id"]: d for d in doctors_data}

    # Build a frontend-friendly JSON object for JavaScript
    formatted_slots = []
    formatted_slots.extend(
        {
            "id": s["id"],
            "doctor_id": s["doctor_id"],
            "doctor": doctor_lookup.get(s["doctor_id"], {}).get(
                "specialization", "Unknown"
            ),
            "date": s["available_date"],
            "start": s["start_time"],
            "end": s["end_time"],
        }
        for s in available_slots
    )
    return render_template("book_appointment.html", slots=formatted_slots)

@app.route("/my-bookings")
@login_required
def my_bookings():
    user_id = session.get("user_id")
    # --- NEW: Include meeting_url in query ---
    appointments = supabase.table("appointments").select(
        "id, appointment_time, status, doctor_id, meeting_url" # Include meeting_url
    ).eq("user_id", user_id).order("appointment_time", desc=True).execute().data

    doctor_lookup = {}
    try:
        if doctor_ids := list({a["doctor_id"] for a in appointments}):
            doctors = supabase.table("doctors").select("id, specialization").in_("id", doctor_ids).execute().data
            doctor_lookup = {d["id"]: d for d in doctors}
    except Exception as e:
        print("Doctor fetch error:", e)

    return render_template("my_bookings.html", appointments=appointments, doctors=doctor_lookup)

# ============================================================
# Routes - Subscription Management (RenalGuard Specific)
# ============================================================

@app.route("/pricing")
def pricing():
    """Display available subscription plans."""
    try:
        # Fetch all active subscription plans from Supabase
        response = supabase.table("subscription_plans").select("*").eq("is_active", True).execute()
        plans = response.data or []
        # Sort by price (optional)
        plans.sort(key=lambda x: float(x["price"]))
    except Exception as e:
        print(f"Error fetching plans: {e}")
        flash("Error loading plans. Please try again later.", "danger")
        plans = [] # Fallback to empty list if fetch fails
    return render_template("pricing.html", plans=plans)

@app.route("/subscribe/<plan_name>", methods=["GET", "POST"])
@login_required
def subscribe(plan_name):
    if session.get("role") != "user":
        flash("Only users can subscribe.", "danger")
        return redirect(url_for("user_dashboard"))

    # Fetch selected plan
    plan_response = supabase.table("subscription_plans").select("*").eq("name", plan_name).execute()
    plan = plan_response.data[0] if plan_response.data else None
    if not plan:
        flash("Plan not found.", "danger")
        return redirect(url_for("user_dashboard"))

    user_id = session["user_id"]

    if request.method == "POST":
        # Create Paystack transaction
        amount_kobo = int(float(plan["price"]) * 100)
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "email": session.get("user_email"), # Ensure email is stored in session during login
            "amount": amount_kobo,
            "callback_url": url_for("verify_payment", _external=True),
            "metadata": {
                "user_id": user_id,
                "plan_id": plan["id"],
                "plan_name": plan["name"]
            }
        }
        response = requests.post("https://api.paystack.co/transaction/initialize", json=data, headers=headers)
        res_data = response.json()
        if res_data.get("status"):
            auth_url = res_data["data"]["authorization_url"]
            return redirect(auth_url)
        else:
            flash("Failed to initialize payment.", "danger")
    return render_template("subscribe_confirm.html", plan=plan, PAYSTACK_PUBLIC_KEY=PAYSTACK_PUBLIC_KEY)

@app.route("/verify_payment")
@login_required
def verify_payment():
    reference = request.args.get("reference")
    if not reference:
        flash("Missing payment reference.", "danger")
        return redirect(url_for("user_dashboard"))

    headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
    response = requests.get(f"https://api.paystack.co/transaction/verify/{reference}", headers=headers)
    res_data = response.json()
    if res_data.get("status") and res_data["data"]["status"] == "success":
        metadata = res_data["data"]["metadata"]
        user_id = metadata["user_id"]
        plan_id = metadata["plan_id"]

        # Record in Supabase
        supabase.table("user_subscriptions").insert({
            "user_id": user_id,
            "plan_id": plan_id,
            "status": "active",
            "start_date": datetime.now(timezone.utc).isoformat()
        }).execute()

        flash("Subscription successful!", "success")
    else:
        flash("Payment verification failed.", "danger")
    return redirect(url_for("user_dashboard"))

@app.route("/user/subscriptions/cancel/<sub_id>", methods=["POST"])
@login_required
def cancel_subscription(sub_id):
    if session.get("role") != "user":
        flash("Unauthorized", "danger")
        return redirect(url_for("user_dashboard"))

    user_id = session.get("user_id")
    # Update status to cancelled
    supabase.table("user_subscriptions").update({"status": "cancelled"}).eq("id", sub_id).eq("user_id", user_id).execute()
    flash("Subscription cancelled.", "info")
    return redirect(url_for("user_dashboard"))

# ============================================================
# Routes - Dashboards (RenalGuard Specific)
# ============================================================

@app.route("/doctor/dashboard")
@login_required
def doctor_dashboard():
    # Check if user is logged in as a doctor
    if session.get("role") != "doctor":
        flash("Access denied. Doctors only.", "danger")
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    doctor_id = session.get("doctor_id") # Assuming this is set during login for doctors
    # If doctor_id isn't in session, fetch it from the doctors table
    if not doctor_id:
        try:
            doc_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
            if doc_data and doc_data:
                doctor_id = doc_data.data["id"]
                session["doctor_id"] = doctor_id # Store for future use in this session
            else:
                flash("Doctor profile not found. Please contact support.", "danger")
                return redirect(url_for("login"))
        except Exception as e:
            print(f"Error fetching doctor ID: {e}")
            flash("Error accessing dashboard. Please try again later.", "danger")
            return redirect(url_for("login"))

    try:
        # Fetch doctor profile details (name, email, bio, specialization from users and doctors tables)
        user_data = supabase.table("users").select("name, email").eq("id", user_id).single().execute().data
        doctor_details = supabase.table("doctors").select("bio, specialization, consultation_fee").eq("id", doctor_id).single().execute().data
        # Combine user and doctor data
        profile_info = {**user_data, **doctor_details}

        # Fetch availability slots for this doctor
        availability_slots = supabase.table("doctor_availability").select("*").eq("doctor_id", doctor_id).order("available_date").execute().data

        # Fetch booked appointments for this doctor
        # --- NEW: Include meeting_url in query ---
        booked_appointments = supabase.table("appointments").select("id, user_id, appointment_time, status, meeting_url").eq("doctor_id", doctor_id).order("appointment_time").execute().data # Include meeting_url

        # Optionally, fetch user details for the booked appointments
        user_ids_needed = [appt["user_id"] for appt in booked_appointments]
        patient_details = {}
        if user_ids_needed:
            users_info = supabase.table("users").select("id, name, email").in_("id", user_ids_needed).execute().data
            patient_details = {u["id"]: u for u in users_info}
    except Exception as e:
        print(f"Error fetching doctor dashboard data: {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("login")) # Or render a partial template

    return render_template(
        "doctor_dashboard.html",
        profile=profile_info,
        availability_slots=availability_slots,
        booked_appointments=booked_appointments,
        patient_details=patient_details
    )

@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if session.get("role") != "user":
        flash("Access denied. Users only.", "danger")
        return redirect(url_for("login"))

    user_id = session.get("user_id")
    try:
        user_data = supabase.table("users").select("name, email").eq("id", user_id).single().execute().data
        # Store email in session for payment verification
        session["user_email"] = user_data["email"]

        # Fetch user's health records if applicable
        records = supabase.table("records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
        # Filter for kidney records if needed
        kidney_records = [r for r in records if 'kidney' in r.get('consultation_type', '').lower()]

        # Fetch user's booked appointments
        # --- NEW: Include meeting_url in query ---
        appointments = supabase.table("appointments").select(
            "id, doctor_id, appointment_time, status, meeting_url" # Include meeting_url
        ).eq("user_id", user_id).order("appointment_time", desc=True).execute().data

        # Fetch doctor details for the appointments
        doctor_ids_needed = [appt["doctor_id"] for appt in appointments]
        doctor_details = {}
        if doctor_ids_needed:
            doctors_info = supabase.table("doctors").select("id, specialization").in_("id", doctor_ids_needed).execute().data
            doctors_lookup = {d["id"]: d for d in doctors_info}
            users_info = supabase.table("users").select("id, name").in_("id", doctor_ids_needed).execute().data
            users_lookup = {u["id"]: u for u in users_info}
            # Combine doctor and user details
            for appt in appointments:
                doc_id = appt["doctor_id"]
                doc_info = doctors_lookup.get(doc_id, {})
                user_info = users_lookup.get(doc_id, {})
                doctor_details[doc_id] = {**doc_info, **user_info} # e.g., {'specialization': 'Cardiology', 'name': 'Dr. Smith'}

        # Prepare data for charts (example remains the same)
        if kidney_records: # Use kidney-specific records
            labels = [r["created_at"][:10] for r in kidney_records]
            scores = [r["health_score"] for r in kidney_records]
        else:
            labels, scores = [], []

        # --- NEW: Fetch user subscriptions ---
        user_subscriptions = supabase.table("user_subscriptions").select(
            "*, subscription_plans(name, price, duration_days, is_free)"
        ).eq("user_id", user_id).execute().data

        # Determine current plan status for UI hints
        current_plan_is_free = True
        if user_subscriptions:
            # Get the most recent active or non-cancelled subscription
            active_subs = [sub for sub in user_subscriptions if sub.get("status") != "cancelled"]
            if active_subs:
                # Sort by start_date descending to get the latest
                latest_sub = sorted(active_subs, key=lambda x: x.get("start_date", ""), reverse=True)[0]
                plan_info = latest_sub.get("subscription_plans", {})
                current_plan_is_free = plan_info.get("is_free", True)

    except Exception as e:
        print(f"Error fetching user dashboard data: {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("login")) # Or render a partial template

    return render_template(
        "user_dashboard.html",
        user=user_data,
        records=kidney_records, # Pass kidney-specific records
        chart_labels=json.dumps(labels),
        chart_scores=json.dumps(scores),
        appointments=appointments, # Pass appointments to the template
        doctor_details=doctor_details, # Pass doctor details for the template
        user_subscriptions=user_subscriptions, # Pass subscriptions to the template
        current_plan_is_free=current_plan_is_free # Pass plan status for UI
    )

@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if session.get("role") != "admin":
        flash("Access denied. Admins only.", "danger")
        return redirect(url_for("login"))

    try:
        # Fetch core counts
        total_users = len(supabase.table("users").select("id").execute().data)
        total_doctors = len(supabase.table("doctors").select("id").execute().data)
        total_admins = len(supabase.table("admins").select("id").execute().data)
        total_records = len(supabase.table("records").select("id").execute().data)
        total_chats = len(supabase.table("chat_logs").select("id").execute().data)
        total_appointments = len(supabase.table("appointments").select("id").execute().data)

        # Fetch counts for different appointment statuses
        appointment_statuses = supabase.table("appointments").select("status").execute().data
        status_counts = {}
        for appt in appointment_statuses:
            status = appt.get("status", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        # Fetch admin details
        admins = supabase.table("admins").select("name, email").execute().data

        # Aggregate data for chart (records by consultation type)
        record_data = supabase.table("records").select("consultation_type").execute().data
        type_count = {}
        for r in record_data:
            # Group by kidney types
            ctype = r.get("consultation_type", "Unknown")
            if 'kidney' in ctype.lower():
                if 'clinical' in ctype.lower():
                    key = 'Kidney Clinical'
                elif 'lifestyle' in ctype.lower():
                    key = 'Kidney Lifestyle'
                else:
                    key = 'Kidney Other'
            else:
                key = 'Other' # Or handle other types if they exist
            type_count[key] = type_count.get(key, 0) + 1

        # Aggregate data for user role chart (if roles are stored in users table)
        user_roles = supabase.table("users").select("role").execute().data
        role_counts = {}
        for user in user_roles:
            role = user.get("role", "user") # Default to 'user' if role is missing
            role_counts[role] = role_counts.get(role, 0) + 1

        # --- Activity Chart Data ---
        # Example: Fetch user registrations over time (last 6 months)
        # Get the date 6 months ago
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=6*30)
        six_months_ago_str = six_months_ago.strftime('%Y-%m-%d')
        # Fetch user creation dates within the last 6 months
        user_registrations_raw = supabase.table("users").select("created_at").gte("created_at", six_months_ago_str).execute().data
        # Extract just the date part and count occurrences
        registration_dates = [user["created_at"][:10] for user in user_registrations_raw]
        from collections import Counter
        registration_counts = Counter(registration_dates)
        # Sort the dates for the chart
        sorted_dates = sorted(registration_counts.keys())
        registration_counts_list = [registration_counts[date] for date in sorted_dates]

        # Example: Fetch appointment counts over time (last 6 months)
        appointment_counts_raw = supabase.table("appointments").select("appointment_time").gte("appointment_time", six_months_ago_str).execute().data
        appointment_dates = [appt["appointment_time"][:10] for appt in appointment_counts_raw]
        appointment_counts = Counter(appointment_dates)
        sorted_appt_dates = sorted(appointment_counts.keys())
        appointment_counts_list = [appointment_counts[date] for date in sorted_appt_dates]

        # Prepare labels for the chart (using the sorted unique dates)
        activity_labels = sorted(set(sorted_dates + sorted_appt_dates)) # Combine and sort unique dates
        # Prepare data for user registrations, aligning with activity_labels
        user_activity_data = [registration_counts.get(date, 0) for date in activity_labels]
        # Prepare data for appointments, aligning with activity_labels
        appt_activity_data = [appointment_counts.get(date, 0) for date in activity_labels]

        # --- Fetch all user subscriptions for admin ---
        user_subscriptions = supabase.table("user_subscriptions").select(
            "id, user_id, status, start_date, end_date, subscription_plans(name, price)"
        ).execute().data
        if user_ids_for_subs := list(
            {sub["user_id"] for sub in user_subscriptions}
        ):
            users_for_subs = supabase.table("users").select("id, name, email").in_("id", user_ids_for_subs).execute().data
            user_lookup_for_subs = {u["id"]: u for u in users_for_subs}
        else:
            user_lookup_for_subs = {}

    except Exception as e:
        print(f"Error fetching admin dashboard data: {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("admin_dashboard")) # Or render a partial template

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_doctors=total_doctors,
        total_admins=total_admins,
        total_records=total_records,
        total_chats=total_chats,
        total_appointments=total_appointments,
        admins=admins,
        record_type_labels=json.dumps(list(type_count.keys())),
        record_type_counts=json.dumps(list(type_count.values())),
        user_role_labels=json.dumps(list(role_counts.keys())),
        user_role_counts=json.dumps(list(role_counts.values())),
        appointment_status_labels=json.dumps(list(status_counts.keys())),
        appointment_status_counts=json.dumps(list(status_counts.values())),
        # Pass activity chart data
        activity_labels=json.dumps(activity_labels),
        user_activity_data=json.dumps(user_activity_data),
        appt_activity_data=json.dumps(appt_activity_data),
        # Pass subscription data for admin
        user_subscriptions=user_subscriptions,
        user_lookup_for_subs=user_lookup_for_subs
    )

def log_activity(user_id, plan_id, activity_type):
    supabase.table("user_subscription_activity").insert({
        "user_id": user_id,
        "plan_id": plan_id,
        "activity_type": activity_type
    }).execute()

# ============================================================
# Routes - Utility
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    # For production use gunicorn (or fly/gunicorn)
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "False") == "True")
