import base64
import importlib
import json
import logging
import os
import pickle
import sqlite3
import tempfile
from datetime import date
from importlib import metadata as importlib_metadata

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('bp_app')

try:
    import cv2
except Exception as exc:
    cv2 = None
    logger.warning('cv2 import failed: %s', exc)

try:
    import librosa
except Exception as exc:
    librosa = None
    logger.warning('librosa import failed: %s', exc)

try:
    import numpy as np
except Exception as exc:
    np = None
    logger.warning('numpy import failed: %s', exc)

try:
    import soundfile as sf
except Exception as exc:
    sf = None
    logger.warning('soundfile import failed: %s', exc)

from flask import Flask, jsonify, render_template, request, send_from_directory

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger('bp_app')

DEEPFACE_AVAILABLE = False
DEEPFACE_IMPORT_ERROR = None
DeepFace = None

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info('DeepFace import succeeded')
except Exception as exc:
    DEEPFACE_IMPORT_ERROR = str(exc)
    logger.exception('DeepFace import failed: %s', exc)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.info('Gemini AI loaded')
except Exception as exc:
    logger.exception('Gemini import failed: %s', exc)
    GEMINI_AVAILABLE = False


def verify_dependency(module_name, package_name=None):
    package_name = package_name or module_name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', None)
        if not version:
            try:
                version = importlib_metadata.version(package_name)
            except Exception:
                version = 'unknown'
        logger.info('Verified dependency %s=%s', package_name, version)
        return version
    except Exception as exc:
        logger.warning('Dependency unavailable: %s (%s)', package_name, exc)
        return None


verify_dependency('flask', 'flask')
verify_dependency('numpy', 'numpy')
verify_dependency('cv2', 'opencv-python-headless')
verify_dependency('librosa', 'librosa')
verify_dependency('soundfile', 'soundfile')
verify_dependency('sklearn', 'scikit-learn')
verify_dependency('pandas', 'pandas')
verify_dependency('joblib', 'joblib')
verify_dependency('requests', 'requests')
verify_dependency('mediapipe', 'mediapipe')
verify_dependency('google.generativeai', 'google-generativeai')
verify_dependency('tensorflow', 'tensorflow')

app = Flask(__name__)

# ─── DB Init ──────────────────────────────
def get_db():
    # Use an absolute path for the DB in Cloud Run/Functions environments
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bp_records.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as _db:
        _db.execute('''
            CREATE TABLE IF NOT EXISTS food_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                log_date      TEXT NOT NULL,
                meal_name     TEXT NOT NULL,
                category      TEXT NOT NULL,
                sodium        TEXT NOT NULL,
                portion       TEXT NOT NULL,
                logged_at     TEXT NOT NULL,
                rating        TEXT DEFAULT '' ,
                rating_reason TEXT DEFAULT ''
            )
        ''')
        # Safe migration for existing rows
        for col in [('rating', "TEXT DEFAULT ''"), ('rating_reason', "TEXT DEFAULT ''")]:
            try:
                _db.execute(f'ALTER TABLE food_log ADD COLUMN {col[0]} {col[1]}')
            except Exception:
                pass
        _db.commit()

init_db()


# ─── Load model & features ────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'logreg_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.pkl')
MODEL_STATS_PATH = os.path.join(BASE_DIR, 'static', 'model_comparison.json')

model = None
FEATURES = None
MODEL_STATS = {}

for path in (MODEL_PATH, FEATURES_PATH):
    if not os.path.exists(path):
        logger.warning('Expected model file not found: %s', path)

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info('Loaded model from %s', MODEL_PATH)
except Exception as exc:
    logger.exception('Failed to load model from %s: %s', MODEL_PATH, exc)

try:
    with open(FEATURES_PATH, 'rb') as f:
        FEATURES = pickle.load(f)
    logger.info('Loaded feature list from %s', FEATURES_PATH)
except Exception as exc:
    logger.exception('Failed to load feature list from %s: %s', FEATURES_PATH, exc)

try:
    with open(MODEL_STATS_PATH, 'r', encoding='utf-8') as f:
        MODEL_STATS = json.load(f)
    logger.info('Loaded model comparison stats from %s', MODEL_STATS_PATH)
except Exception as exc:
    logger.exception('Failed to load model comparison stats from %s: %s', MODEL_STATS_PATH, exc)

STAGE_INFO = {
    0: {
        'label': 'Normal',
        'color': '#22c55e',
        'badge': 'normal',
        'description': 'Blood pressure is within the normal range. No hypertension detected.',
        'recommendations': [
            'Maintain a healthy diet low in sodium and saturated fats',
            'Exercise at least 150 minutes per week at moderate intensity',
            'Monitor blood pressure annually during routine check-ups',
            'Maintain a healthy body weight (BMI 18.5–24.9)',
            'Limit alcohol consumption and avoid tobacco products',
        ],
        'urgency': '✅ Routine monitoring recommended',
        'urgency_color': '#22c55e',
    },
    1: {
        'label': 'Stage-1 Hypertension',
        'color': '#f59e0b',
        'badge': 'stage1',
        'description': 'Early-stage hypertension. Lifestyle modifications are strongly recommended.',
        'recommendations': [
            'Adopt the DASH (Dietary Approaches to Stop Hypertension) diet',
            'Reduce sodium intake to less than 2,300 mg/day',
            'Increase physical activity to 30–60 minutes most days',
            'Schedule a follow-up appointment within 1–3 months',
            'Monitor blood pressure at home regularly (twice daily)',
            'Discuss medication options with your healthcare provider',
        ],
        'urgency': '🗓️ Consult your doctor within 1–3 months',
        'urgency_color': '#f59e0b',
    },
    2: {
        'label': 'Stage-2 Hypertension',
        'color': '#ef4444',
        'badge': 'stage2',
        'description': 'Significant hypertension detected. Medical intervention required.',
        'recommendations': [
            'Seek medical consultation as soon as possible',
            'Medication therapy is likely required — consult your doctor',
            'Strictly limit sodium to less than 1,500 mg/day',
            'Begin a supervised cardiac rehabilitation program',
            'Monitor blood pressure daily and keep a detailed log',
            'Reduce stress through meditation and breathing exercises',
            'Avoid strenuous physical activity until evaluated by a doctor',
        ],
        'urgency': '⚠️ See a doctor within 1 week',
        'urgency_color': '#ef4444',
    },
    3: {
        'label': 'Hypertensive Crisis',
        'color': '#7f1d1d',
        'badge': 'crisis',
        'description': '⚠️ Critical hypertension level detected. Immediate medical attention required.',
        'recommendations': [
            '🚨 SEEK EMERGENCY MEDICAL CARE IMMEDIATELY',
            'Call emergency services (112 / 911) or go to the nearest ER',
            'Do NOT take any new medications without emergency guidance',
            'Sit calmly and avoid all physical exertion',
            'If on medication, report last dose and timing to emergency staff',
            'Inform medical personnel of all current medications',
        ],
        'urgency': '🚨 Seek emergency care IMMEDIATELY',
        'urgency_color': '#ef4444',
    }
}

# Feature importance proxy (weights from LogReg or named importance)
FEATURE_LABELS = {
    'Gender': 'Gender',
    'Age_Group': 'Age Group',
    'Family_History': 'Family History',
    'Patient_Status': 'Patient Status',
    'Take_Medication': 'Medication',
    'Time_Since_Diagnosis': 'Time Since Dx',
    'Symptom_Severity': 'Symptom Severity',
    'Shortness_of_Breath': 'Shortness of Breath',
    'Visual_Changes': 'Visual Changes',
    'Nosebleeds': 'Nosebleeds',
    'Systolic_BP': 'Systolic BP',
    'Diastolic_BP': 'Diastolic BP',
    'Controlled_Diet': 'Controlled Diet',
    'BMI_Category': 'BMI Category',
    'Diabetes': 'Diabetes',
    'Cholesterol_Level': 'Cholesterol',
    'Heart_Rate_Category': 'Heart Rate',
    'Exercise_Frequency': 'Exercise Freq.',
}


@app.route('/health')
def health():
    def package_version(name):
        try:
            return importlib_metadata.version(name)
        except Exception:
            return 'unknown'

    model_files = [
        str(MODEL_PATH),
        str(FEATURES_PATH),
        str(MODEL_STATS_PATH),
    ]
    return jsonify({
        'python_version': os.sys.version.split()[0],
        'deepface_installed': DEEPFACE_AVAILABLE,
        'deepface_error': DEEPFACE_IMPORT_ERROR,
        'tensorflow_version': package_version('tensorflow'),
        'opencv_version': package_version('opencv-python-headless') or package_version('opencv-python'),
        'mediapipe_version': package_version('mediapipe'),
        'model_files_found': {path: os.path.exists(path) for path in model_files},
    })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        feature_values = [
            int(data['gender']),
            int(data['age_group']),
            int(data['family_history']),
            int(data['patient_status']),
            int(data['take_medication']),
            int(data['time_since_diagnosis']),
            int(data['symptom_severity']),
            int(data['shortness_of_breath']),
            int(data['visual_changes']),
            int(data['nosebleeds']),
            int(data['systolic_bp']),
            int(data['diastolic_bp']),
            int(data['controlled_diet']),
            int(data['bmi_category']),
            int(data['diabetes']),
            int(data['cholesterol_level']),
            int(data['heart_rate_category']),
            int(data['exercise_frequency']),
        ]
        features = np.array([feature_values])

        stage   = int(model.predict(features)[0])
        probas  = model.predict_proba(features)[0]
        confidence = round(float(probas[stage]) * 100, 1)

        info = STAGE_INFO[stage]

        # Build radar data: normalize each input to 0–1 scale for risk profile
        max_vals = [1, 4, 1, 1, 1, 3, 2, 1, 1, 1, 4, 3, 1, 3, 1, 2, 3, 5]
        radar_vals = [round(feature_values[i] / max_vals[i], 2) for i in range(18)]

        return jsonify({
            'stage':            stage,
            'label':            info['label'],
            'color':            info['color'],
            'badge':            info['badge'],
            'confidence':       confidence,
            'description':      info['description'],
            'recommendations':  info['recommendations'],
            'urgency':          info['urgency'],
            'urgency_color':    info['urgency_color'],
            'all_probabilities': {
                'Normal':  round(float(probas[0]) * 100, 1),
                'Stage-1': round(float(probas[1]) * 100, 1),
                'Stage-2': round(float(probas[2]) * 100, 1),
                'Crisis':  round(float(probas[3]) * 100, 1),
            },
            'radar': {
                'labels': list(FEATURE_LABELS.values()),
                'values': radar_vals,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model_stats')
def model_stats():
    return jsonify(MODEL_STATS)


# ─── Food Log Routes ─────────────────────
FOOD_SUGGESTIONS = {
    0: {
        'stage': 'Normal',
        'color': '#22c55e',
        'eat': [
            '🥦 Broccoli, spinach, kale — potassium-rich greens',
            '🍌 Bananas, oranges, avocado — natural BP regulators',
            '🐟 Salmon, mackerel — omega-3 fatty acids',
            '🥛 Low-fat yogurt & milk — calcium for vessels',
            '🌾 Oats, brown rice, quinoa — whole grains',
            '🥜 Almonds, walnuts — healthy fats',
            '🫐 Blueberries, strawberries — antioxidants',
        ],
        'avoid': [
            '🧂 Excessive salt (> 2,300 mg sodium/day)',
            '🍟 Deep-fried fast foods',
            '🥤 Sugary sodas and energy drinks',
        ],
        'tip': 'Maintain a balanced diet with variety. Keep sodium under 2,300 mg/day.',
    },
    1: {
        'stage': 'Stage-1 Hypertension',
        'color': '#f59e0b',
        'eat': [
            '🥬 DASH diet: leafy greens, beets, celery',
            '🍠 Sweet potato — high potassium (941 mg each)',
            '🧄 Garlic — natural vasodilator',
            '🐟 Fatty fish 2× per week',
            '🫘 Lentils, kidney beans — plant protein + fibre',
            '🍋 Lemon water — supports BP balance',
            '🌶️ Cayenne pepper — capsaicin helps vessels relax',
        ],
        'avoid': [
            '🧂 Limit sodium < 2,000 mg/day',
            '🍕 Processed meats (bacon, sausage, deli meat)',
            '🥫 Canned soups & sauces (very high sodium)',
            '🍺 Alcohol (limit to 1 drink/day)',
            '☕ Excess caffeine (> 2 cups/day)',
        ],
        'tip': 'Follow the DASH diet. Reduce sodium intake and increase potassium-rich foods.',
    },
    2: {
        'stage': 'Stage-2 Hypertension',
        'color': '#ef4444',
        'eat': [
            '🥗 Raw salads with olive oil dressing (no salt)',
            '🥝 Kiwi — studies show 3/day reduces systolic BP',
            '🫒 Extra virgin olive oil — monounsaturated fats',
            '🍇 Dark grapes — resveratrol for heart health',
            '🌿 Flaxseed — alpha-linolenic acid reduces BP',
            '🥛 Skim milk — calcium + low fat',
            '🫚 Beetroot juice — nitrates dilate blood vessels',
        ],
        'avoid': [
            '🚫 Sodium < 1,500 mg/day (very strict limit)',
            '🥩 Red meat and saturated fats',
            '🍰 Sweets, pastries, trans fats',
            '🍺 Alcohol — completely if possible',
            '🧃 Fruit juices (high sugar spike)',
            '🍔 All fast food and restaurant meals',
        ],
        'tip': '⚠️ Strict low-sodium diet required. Consult a dietitian alongside your doctor.',
    },
    3: {
        'stage': 'Hypertensive Crisis',
        'color': '#7f1d1d',
        'eat': [
            '🚑 Follow ONLY hospital/doctor dietary instructions',
            '💧 Small sips of water only if advised',
            '🥣 If permitted: plain oats, plain boiled rice',
            '🍌 A single banana if potassium is low (ask doctor)',
        ],
        'avoid': [
            '🚨 ANY high-sodium food — absolutely forbidden',
            '🚨 Caffeine, alcohol, stimulants completely',
            '🚨 Do NOT self-medicate with supplements',
            '🚨 Avoid all food until medically assessed',
        ],
        'tip': '🚨 EMERGENCY: Seek immediate medical care. Do not manage diet independently.',
    },
}

RATING_LABELS = {
    'Excellent': {'icon': '✅', 'color': '#22c55e'},
    'Good':      {'icon': '👍', 'color': '#84cc16'},
    'Caution':   {'icon': '⚠️', 'color': '#f59e0b'},
    'Dangerous': {'icon': '🚨', 'color': '#ef4444'},
}

def gemini_rate_meal(meal_name, sodium, stage, category):
    """Ask Gemini to rate a meal for a hypertension patient. Returns (rating, reason)."""
    import requests as _req
    stage_name = STAGE_NAMES.get(stage, 'Normal Blood Pressure')
    prompt = f"""You are a clinical dietitian evaluating meals for a hypertension patient.

Patient: {stage_name}
Meal logged: "{meal_name}" (Category: {category}, Sodium level: {sodium})

Rate this specific meal for this patient. Reply with ONLY a valid JSON object:
{{"rating": "Excellent" | "Good" | "Caution" | "Dangerous", "reason": "one short sentence (max 12 words) explaining why"}}

Rating guide:
- Excellent: heart-healthy, low-sodium, anti-hypertensive properties
- Good: generally healthy, acceptable for this stage
- Caution: moderate concern, occasional consumption only
- Dangerous: high sodium / high fat / contraindicated for this stage
"""
    try:
        r = _req.post(
            GEMINI_URL,
            headers={'Content-Type': 'application/json', 'X-goog-api-key': GEMINI_API_KEY},
            json={'contents': [{'parts': [{'text': prompt}]}],
                  'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 80}},
            timeout=12,
        )
        r.raise_for_status()
        text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        text = text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(text)
        rating = parsed.get('rating', 'Good')
        if rating not in RATING_LABELS:
            rating = 'Good'
        return rating, parsed.get('reason', '')
    except Exception:
        # Fallback: derive from sodium level alone
        fallback = {'Low': 'Good', 'Medium': 'Caution', 'High': 'Dangerous'}.get(sodium, 'Caution')
        return fallback, 'Rated based on sodium level (AI unavailable)'

@app.route('/food-log', methods=['POST'])
def add_food_log():
    data = request.get_json()
    try:
        from datetime import datetime
        meal   = data['meal_name'].strip()
        cat    = data['category']
        sodium = data['sodium']
        stage  = int(data.get('stage', 0))

        # Get Gemini rating
        rating, reason = gemini_rate_meal(meal, sodium, stage, cat)

        with get_db() as conn:
            conn.execute(
                'INSERT INTO food_log (log_date, meal_name, category, sodium, portion, logged_at, rating, rating_reason) VALUES (?,?,?,?,?,?,?,?)',
                (str(date.today()), meal, cat, sodium, data['portion'],
                 datetime.now().strftime('%H:%M'), rating, reason)
            )
            conn.commit()
        return jsonify({'ok': True, 'rating': rating, 'reason': reason})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/food-log', methods=['GET'])
def get_food_log():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM food_log WHERE log_date = ? ORDER BY id DESC',
            (str(date.today()),)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyAjScSXNKeIBctP1thxbe8J-Rott7oSs04')
GEMINI_URL = (
    'https://generativelanguage.googleapis.com/v1beta/models/'
    'gemini-2.0-flash:generateContent'
)

STAGE_NAMES = {
    0: 'Normal Blood Pressure',
    1: 'Stage-1 Hypertension (130-139/80-89 mmHg)',
    2: 'Stage-2 Hypertension (140+/90+ mmHg)',
    3: 'Hypertensive Crisis (180+/120+ mmHg)',
}

@app.route('/food-suggestions')
def food_suggestions():
    import requests as _req

    try:
        stage    = int(request.args.get('stage', 0))
        diabetes = request.args.get('diabetes', '0') == '1'
        bmi      = request.args.get('bmi', '1')
        bmi_labels = {0:'Underweight', 1:'Normal BMI', 2:'Overweight', 3:'Obese'}
        if stage not in STAGE_NAMES:
            stage = 0
    except ValueError:
        stage = 0

    stage_name = STAGE_NAMES[stage]
    bmi_label  = bmi_labels.get(int(bmi), 'Normal BMI')
    extras = []
    if diabetes:
        extras.append('Type-2 Diabetes')
    if int(bmi) >= 2:
        extras.append(bmi_label)
    context = ', '.join(extras) if extras else 'no additional conditions'

    prompt = f"""You are a clinical dietitian specializing in hypertension and cardiovascular health.

Patient profile:
- Hypertension stage: {stage_name}
- Additional conditions: {context}

Generate personalised daily dietary advice. Reply with ONLY a valid JSON object (no markdown, no explanation):
{{
  "eat": ["emoji item — brief reason", ...],
  "avoid": ["emoji item — brief reason", ...],
  "tip": "one sentence of practical dietary advice"
}}
Rules:
- "eat" must have exactly 7 items, "avoid" exactly 5 items
- Each item starts with a relevant food emoji
- Advice must reflect the patient's exact stage and co-morbidities
- For Crisis stage, reflect emergency dietary context
"""

    try:
        resp = _req.post(
            GEMINI_URL,
            headers={
                'Content-Type': 'application/json',
                'X-goog-api-key': GEMINI_API_KEY,
            },
            json={
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {'temperature': 0.4, 'maxOutputTokens': 600},
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw  = resp.json()
        text = raw['candidates'][0]['content']['parts'][0]['text'].strip()
        text = text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(text)

        return jsonify({
            'stage':  stage_name,
            'color':  FOOD_SUGGESTIONS[stage]['color'],
            'eat':    parsed.get('eat',   FOOD_SUGGESTIONS[stage]['eat']),
            'avoid':  parsed.get('avoid', FOOD_SUGGESTIONS[stage]['avoid']),
            'tip':    parsed.get('tip',   FOOD_SUGGESTIONS[stage]['tip']),
            'source': 'gemini',
        })

    except Exception as e:
        import logging
        logging.warning(f'Gemini API error: {e}')
        fallback = FOOD_SUGGESTIONS[stage].copy()
        fallback['source'] = 'static'
        # Include the error type so frontend can show a meaningful message
        err_str = str(e)
        if '429' in err_str or 'quota' in err_str.lower() or 'rate' in err_str.lower():
            fallback['error_msg'] = 'Rate limit reached — showing curated suggestions'
        elif 'timeout' in err_str.lower():
            fallback['error_msg'] = 'Gemini timed out — showing curated suggestions'
        else:
            fallback['error_msg'] = 'Gemini suggestion loading failed — showing curated suggestions'
        return jsonify(fallback)


# ── Facial Hypertension Screening ───────────────────────────────────────────
@app.route('/facial-predict', methods=['POST'])
def facial_predict():
    print(">>> RECEIVED REQUEST ON /facial-predict")
    import requests as _req, base64, json as _json

    data = request.get_json()
    img_b64 = data.get('image', '')           # base64 data-URL or raw base64
    if img_b64.startswith('data:'):
        img_b64 = img_b64.split(',', 1)[1]   # strip "data:image/jpeg;base64,"

    prompt = """You are a clinical AI expert trained in facial psychophysiology and cardiovascular assessment.

Analyse this facial photograph carefully for the following:

1. HYPERTENSION SIGNS: Facial flushing/redness, periorbital puffiness, scleral redness, facial swelling, pallor, yellowish tint
2. EMOTION: Identify the primary visible emotion (Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised, Anxious, Stressed, Calm)
3. STRESS LEVEL: Based on facial tension, brow furrows, jaw clenching, eye strain, overall fatigue signs
4. BLOOD PRESSURE ESTIMATE: Based on visible vasodilation/vasoconstriction, facial colour changes, and visible tension cues

Reply ONLY with a valid JSON object (no markdown, no explanation):
{
  "risk": "Low" | "Moderate" | "High" | "Critical",
  "signs": ["each observed hypertension sign"],
  "emotion": "primary emotion from the list above",
  "emotion_detail": "brief description of what facial cues indicate this emotion",
  "stress_level": "Low" | "Moderate" | "High" | "Severe",
  "stress_cues": ["facial cues indicating stress level"],
  "bp_estimate": {
    "systolic": "estimated systolic in mmHg as a range e.g. 120-130",
    "diastolic": "estimated diastolic in mmHg as a range e.g. 80-90",
    "category": "Normal | Elevated | Stage 1 | Stage 2"
  },
  "confidence": "Low" | "Moderate" | "High",
  "summary": "2-sentence overall assessment",
  "advice": "one actionable recommendation",
  "disclaimer": "This is a non-invasive visual estimate only, not a medical diagnosis. Consult a doctor for accurate BP measurement."
}

Risk guide: Low=no signs, Moderate=1-2 minor signs, High=2-3 signs, Critical=multiple severe signs
Stress guide: Low=relaxed, Moderate=some tension, High=clear strain, Severe=extreme tension/distress"""

            
    local_result = {
        "emotion": "Neutral",
        "stress_level": "Moderate",
        "stress_cues": ["Local analysis pending"],
        "risk": "Low",
        "signs": [],
        "emotion_detail": "DeepFace analysis pending",
        "bp_estimate": {
            "systolic": "120-130",
            "diastolic": "80-90",
            "category": "Normal"
        },
        "confidence": "Moderate",
        "summary": "Visual analysis completed with a local fallback estimate.",
        "advice": "Take a moment to relax and re-check if needed.",
        "disclaimer": "This is a non-invasive visual estimate only, not a medical diagnosis. Consult a doctor for accurate BP measurement.",
        "ok": True,
    }

    if DEEPFACE_AVAILABLE:
        try:
            img_data = base64.b64decode(img_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name

            try:
                objs = DeepFace.analyze(img_path=tmp_path, actions=['emotion'], enforce_detection=False)
                if objs:
                    res = objs[0]
                    dominant_emotion = str(res.get('dominant_emotion', 'neutral')).lower()
                    emotion = dominant_emotion.capitalize()
                    stress_map = {
                        'angry': 'High',
                        'fear': 'High',
                        'sad': 'High',
                        'neutral': 'Moderate',
                        'happy': 'Low',
                        'surprise': 'Low',
                        'disgust': 'Moderate',
                    }
                    stress_level = stress_map.get(dominant_emotion, 'Moderate')
                    local_result.update({
                        "emotion": emotion,
                        "stress_level": stress_level,
                        "stress_cues": [f"DeepFace detected {emotion}"],
                        "emotion_detail": f"The detected expression looks {dominant_emotion}.",
                        "confidence": "High" if stress_level != 'Moderate' else 'Moderate',
                        "summary": f"DeepFace detected a {dominant_emotion} expression and estimated a {stress_level.lower()} stress level.",
                    })
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as exc:
            logger.exception('DeepFace analysis failed: %s', exc)
            local_result['stress_cues'] = [f"DeepFace analysis failed: {exc}"]
            local_result['summary'] = 'DeepFace analysis failed; using local fallback estimate.'
    else:
        logger.warning('DeepFace unavailable: %s', DEEPFACE_IMPORT_ERROR or 'unknown error')
        local_result['stress_cues'] = [f"DeepFace unavailable: {DEEPFACE_IMPORT_ERROR or 'unknown error'}"]
        local_result['summary'] = 'DeepFace import failed; using local fallback estimate.'

    # --- 2. HYBRID ENHANCEMENT WITH GEMINI ---
    try:
        print(f">>> Sending prompt to Gemini: {prompt[:100]}...")
        resp = _req.post(
            GEMINI_URL,
            headers={'Content-Type': 'application/json', 'X-goog-api-key': GEMINI_API_KEY},
            json={
                'contents': [{
                    'parts': [
                        {'text': prompt},
                        {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}},
                    ]
                }],
                'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 400},
            },
            timeout=20,
        )
        
        if resp.status_code == 200:
            raw = resp.json()
            text = raw['candidates'][0]['content']['parts'][0]['text'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            gemini_data = _json.loads(text)
            
            # Merge Gemini's health insights with Local's emotional detection
            gemini_data['emotion'] = local_result['emotion']
            gemini_data['stress_level'] = local_result['stress_level']
            gemini_data['ok'] = True
            return jsonify(gemini_data)
        else:
            print(f">>> Gemini fallback triggered. Status: {resp.status_code}")
            local_result['ok'] = True
            local_result['advice'] = "Local analysis performed. Gemini advice unavailable."
            local_result['disclaimer'] = "AI results are approximate. DeepFace local mode."
            return jsonify(local_result)

    except Exception as e:
        import logging
        import traceback
        logging.warning(f'Facial predict error: {e}')
        print(f">>> FULL TRACEBACK:\n{traceback.format_exc()}")
        # Return local results even if Gemini fails
        local_result['ok'] = True
        local_result['advice'] = "Local analysis successful. Cloud services currently disconnected."
        return jsonify(local_result)


@app.route('/voice-predict', methods=['POST'])
def voice_predict():
    print(">>> RECEIVED REQUEST ON /voice-predict")
    try:
        data = request.get_json()
        audio_b64 = data.get('audio', '')
        if not audio_b64:
            return jsonify({'ok': False, 'error': 'No audio data received'})

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as decode_err:
            return jsonify({'ok': False, 'error': f'Failed to decode audio: {decode_err}'})
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            # Load with librosa - sr=None to preserve original sample rate
            # We use try/except block for librosa loading as it can fail on malformed buffers
            try:
                y, sr = librosa.load(tmp_path, sr=None)
                if len(y) == 0:
                    return jsonify({'ok': False, 'error': 'Audio file is empty or silent'})
            except Exception as librosa_err:
                print(f">>> Librosa Load Error: {librosa_err}")
                return jsonify({'ok': False, 'error': 'Could not process audio format. Please try again.'})
            
            # 1. Pitch Detection (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            # Find mean pitch for non-zero magnitudes above a threshold
            pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.15]
            avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

            # 2. Energy (Loudness/RMS)
            rms = librosa.feature.rms(y=y)
            avg_energy = float(np.mean(rms))

            # 3. Zero Crossing Rate (Spectral density / Voice Shakiness)
            zcr = librosa.feature.zero_crossing_rate(y)
            avg_zcr = float(np.mean(zcr))

            # IMPROVED HEURISTIC STRESS MAPPING
            # Stress often correlates with higher pitch, higher energy (tension), 
            # and higher ZCR (breathiness/shakiness).
            stress_score = 0
            
            # ZCR heuristic (Shakiness)
            if avg_zcr > 0.12: stress_score += 2
            elif avg_zcr > 0.08: stress_score += 1
            
            # Energy heuristic (Tension)
            if avg_energy > 0.06: stress_score += 2
            elif avg_energy > 0.03: stress_score += 1
            
            # Pitch heuristic (Emotional arousal)
            if avg_pitch > 220: stress_score += 1 # High pitch arousal
            if avg_pitch > 0 and avg_pitch < 80: stress_score += 1 # Very low/tense voice

            # Determine level and advice
            if stress_score >= 4:
                stress_level = "High"
                advice = "Your vocal profile indicates significant physiological tension. Consider a 2-minute box breathing exercise and check your blood pressure manually."
                recommended_stage = 2 # Map to Stage 2 for diet suggestions
            elif stress_score >= 1:
                stress_level = "Moderate"
                advice = "Moderate vocal stress detected. This can sometimes elevate short-term blood pressure. Hydration and a brief walk might help."
                recommended_stage = 1 # Map to Stage 1
            else:
                stress_level = "Low"
                advice = "Vocal features appear relaxed and stable. Your voice indicates a healthy level of calm."
                recommended_stage = 0 # Normal

            return jsonify({
                'ok': True,
                'stress_level': stress_level,
                'recommended_stage': recommended_stage,
                'features': {
                    'pitch': round(avg_pitch, 1),
                    'energy': round(avg_energy, 4),
                    'zcr': round(avg_zcr, 4)
                },
                'advice': advice
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        import traceback
        print(f">>> Voice predict fatal error: {e}")
        print(traceback.format_exc())
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
