from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, date, timedelta
from models import db, CareLogEntry, User, Media, AnalysisEntry, AACTrialEntry, Notice, NoticeReply, NoticeRead
from flask_bcrypt import Bcrypt
from utils import sync_to_google_sheets
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from sqlalchemy.sql import text
from sqlalchemy import func
from collections import defaultdict
import bleach
from flask import abort
from flask import make_response


app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.abspath('instance/care_log_v1.sqlite')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads_test')
app.config['TESTING'] = True
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_super_secret_key'

CORS(app, resources={r"/*": {"origins": ["https://zapier.wwave.com.au"], "supports_credentials": True}})


# ---- Allowed HTML for Analysis entries (used when saving) ----
ALLOWED_TAGS = [
    'p','br','strong','b','em','i','u','ul','ol','li','blockquote',
    'h2','h3','h4','h5','h6','a','table','thead','tbody','tr','th','td'
]
ALLOWED_ATTRS = {
    'a': ['href', 'title', 'target', 'rel'],
    'td': ['colspan', 'rowspan'],
    'th': ['colspan', 'rowspan'],
    # Remove '*' if you don't want inline styles saved
    '*': ['style']
}

PROMPTING_LEVELS = [
    "Independent",
    "Expectant/thoughtful pause",
    "Facial expression",
    "Body language",
    "Observation/comment",
    "Visual prompt/gestural cue",
]


SLEEP_MAP = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Excellent": 4,
}

COMM_MAP = {
    "No communication": 0,
    "Passive communication": 1,
    "Minimal communication, not directed at others": 2,
    "Directed communication without back-and-forth": 3,
    "Back and forth communication": 4,
}

def get_communication_score(value, value_type):
    """
    Get communication score from value or value_type, handling partial matches and numeric values.
    Returns score (0-4) or None if not found.
    """
    # First, try to parse as numeric (value field might be "1", "2", "3", "4", "5")
    if value:
        try:
            val_int = int(value)
            # Map 1-5 to 0-4 scale (since your entries use 1-5 but COMM_MAP uses 0-4)
            if 1 <= val_int <= 5:
                return val_int - 1  # Convert 1-5 to 0-4
        except (ValueError, TypeError):
            pass
    
    # Try exact match on value
    if value:
        value_str = str(value).strip()
        if value_str in COMM_MAP:
            return COMM_MAP[value_str]
    
    # Try exact match on value_type
    if value_type:
        value_type_str = str(value_type).strip()
        if value_type_str in COMM_MAP:
            return COMM_MAP[value_type_str]
    
    # Try partial match - check if value_type starts with any COMM_MAP key
    if value_type:
        value_type_lower = str(value_type).strip().lower()
        for key, score in COMM_MAP.items():
            if value_type_lower.startswith(key.lower()):
                return score
    
    # Try partial match on value
    if value:
        value_lower = str(value).strip().lower()
        for key, score in COMM_MAP.items():
            if value_lower.startswith(key.lower()):
                return score
    
    return None

MELTDOWN_MAP = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
}

MENSTRUAL_MAP = {
    "Spotting": 1,
    "Medium": 2,
    "Heavy": 3,
}
MENSTRUAL_INVERSE = {v: k for k, v in MENSTRUAL_MAP.items()}

WALKING_MAP = {
    "Needs Help": 0,
    "Independent": 1,
}

MOTOR_MAP = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
}

POSTURE_KEYWORD_SCORES = [
    ("excellent", 4),
    ("great", 4),
    ("upright", 3),
    ("tall", 3),
    ("good", 3),
    ("neutral", 2),
    ("ok", 2),
    ("fair", 2),
    ("supported", 2),
    ("lean", 1),
    ("leans", 1),
    ("slouch", 1),
    ("hunch", 1),
    ("bent", 1),
    ("poor", 1),
    ("collapsed", 0),
]


def current_user_is_admin():
    u = User.query.filter_by(username=session.get('carer_id')).first()
    return bool(u and u.is_admin)

def can_edit_analysis(entry):
    return entry.carer_id == session.get('carer_id') or current_user_is_admin()


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_ml(value):
    """
    Extract milliliters from fluid intake value.
    Handles both numeric codes (1-5) and actual ml values (50ml, 100ml, etc.)
    """
    if not value:
        return 0.0
    
    # Map numeric codes to actual milliliters (from dropdown: 1=50ml, 2=100ml, 3=200ml, 4=300ml, 5=500ml)
    fluid_code_map = {
        "1": 50.0,
        "2": 100.0,
        "3": 200.0,
        "4": 300.0,
        "5": 500.0
    }
    
    value_str = str(value).strip()
    
    # Check if it's a code (1-5)
    if value_str in fluid_code_map:
        return fluid_code_map[value_str]
    
    # Otherwise, try to extract digits from text like "50ml", "100ml", etc.
    digits = ''.join(ch for ch in value_str if ch.isdigit() or ch == '.')
    try:
        ml_value = float(digits) if digits else 0.0
        # If the extracted value is very small (< 10), it might be a code, so check the map
        if ml_value < 10 and str(int(ml_value)) in fluid_code_map:
            return fluid_code_map[str(int(ml_value))]
        return ml_value
    except ValueError:
        return 0.0


def _posture_score(*texts):
    """
    Extract posture score from text.
    Handles Health / Medical posture labels and keyword matching.
    """
    for text in texts:
        if not text:
            continue
        text = str(text).strip()
        if not text:
            continue
        
        # Check for Health / Medical posture labels first
        lowered = text.lower()
        if "posture - good/upright" in lowered or "posture - good" in lowered:
            return 3.0
        if "posture - supported/neutral" in lowered or "posture - supported" in lowered or "posture - neutral" in lowered:
            return 2.0
        if "posture - leaning/needs support" in lowered or "posture - leaning" in lowered or "posture - needs support" in lowered:
            return 1.0
        if "posture - collapsed" in lowered:
            return 0.0
        
        # Try numeric value
        try:
            return float(text)
        except ValueError:
            pass
        
        # Try keyword matching
        for keyword, score in POSTURE_KEYWORD_SCORES:
            if keyword in lowered:
                return score
    return None


def _posture_label(score):
    if score is None:
        return None
    if score >= 3.5:
        return "Excellent"
    if score >= 2.5:
        return "Good / Upright"
    if score >= 1.5:
        return "Supported / Neutral"
    if score >= 0.5:
        return "Leaning / Needs Support"
    return "Collapsed"


def format_health_summary(health_details):
    """Format health summary: '3 (Dribbling, Supplements, Raynauds)'"""
    if not health_details:
        return None
    
    # Categorize health events (excluding posture which is shown separately)
    categories = []
    for detail in health_details:
        detail_lower = str(detail).lower()
        if 'posture' in detail_lower:
            continue  # Skip posture, shown separately
        elif 'dribbling' in detail_lower:
            categories.append('Dribbling')
        elif 'raynauds' in detail_lower:
            categories.append('Raynauds')
        elif 'bloating' in detail_lower or 'bloated' in detail_lower:
            categories.append('Bloating')
        elif 'medication' in detail_lower or 'supplement' in detail_lower:
            categories.append('Medication/Supplements')
        elif 'doctor' in detail_lower or 'hospital' in detail_lower or 'appointment' in detail_lower:
            categories.append('Medical Appointments')
        elif detail.strip() and detail.strip() != 'Other':
            # Use the actual value if it's meaningful
            categories.append(detail.strip())
    
    # Get unique categories (preserves order)
    seen = set()
    unique_cats = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            unique_cats.append(cat)
    
    count = len(health_details)
    
    if unique_cats:
        # Show top 3 categories
        display_cats = unique_cats[:3]
        return f"{count} ({', '.join(display_cats)})"
    return f"{count} events"


def format_health_tooltip(health_details):
    """Format detailed health tooltip for hover"""
    if not health_details:
        return "No health events"
    
    # Group by type (excluding posture)
    grouped = {}
    for detail in health_details:
        detail_str = str(detail).strip()
        detail_lower = detail_str.lower()
        if 'posture' in detail_lower:
            continue  # Skip posture
        if detail_str and detail_str != 'Other':
            grouped[detail_str] = grouped.get(detail_str, 0) + 1
    
    if not grouped:
        return "No health events (excluding posture)"
    
    lines = []
    for event, count in sorted(grouped.items()):
        if count > 1:
            lines.append(f"• {event} ({count}x)")
        else:
            lines.append(f"• {event}")
    
    # Limit to 10 items to avoid huge tooltips
    display_lines = lines[:10]
    if len(lines) > 10:
        display_lines.append(f"... and {len(lines) - 10} more")
    
    return "<div style='text-align: left; max-width: 300px;'>" + "<br>".join(display_lines) + "</div>"


def format_food_summary(food_types, food_count):
    """Format food summary: '2: Protein, Vegetables, Fruit'"""
    if not food_types or food_count == 0:
        return None
    
    types_list = sorted(list(food_types))[:5]  # Max 5 types
    return f"{food_count}: {', '.join(types_list)}"


def format_food_tooltip(food_details):
    """Format detailed food tooltip for hover"""
    if not food_details:
        return "No food entries"
    
    # Count occurrences
    food_counts = {}
    for food in food_details:
        food_str = str(food).strip()
        if food_str:
            food_counts[food_str] = food_counts.get(food_str, 0) + 1
    
    if not food_counts:
        return "No food entries"
    
    lines = []
    for food, count in sorted(food_counts.items()):
        if count > 1:
            lines.append(f"• {food} ({count}x)")
        else:
            lines.append(f"• {food}")
    
    return "<div style='text-align: left; max-width: 300px;'>" + "<br>".join(lines) + "</div>"


def build_daily_features(start_date=None, end_date=None):
    """Aggregate care log entries into per-day feature records."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=29)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    entries = (
        CareLogEntry.query
        .filter(
            CareLogEntry.activity_datetime >= start_dt,
            CareLogEntry.activity_datetime <= end_dt,
        )
        .order_by(CareLogEntry.activity_datetime.asc())
        .all()
    )

    daily = defaultdict(lambda: {
        "toilet_attempts": 0,
        "toilet_success": 0,
        "accidents": 0,
        "mood_values": [],
        "sleep_scores": [],
        "communication_scores": [],
        "meltdown_scores": [],
        "menstrual_levels": [],
        "fluid_ml": 0.0,
        "food_types": set(),
        "food_details": [],
        "health_events": 0,
        "health_details": [],
        "walking_scores": [],
        "motor_scores": [],
        "notes": [],
        "posture_scores": [],
    })

    for entry in entries:
        day = entry.activity_datetime.date()
        bucket = daily[day]
        name = entry.activity_name or ""
        value = (entry.value or "").strip()
        value_type = (entry.value_type or "").strip()
        notes = (entry.notes or "").strip()

        if name == "Toilet Tries":
            bucket["toilet_attempts"] += 1
            # Check for success: numeric values 2, 3, 4 OR text starting with "yes"
            is_success = False
            if value:
                value_str = str(value).strip().lower()
                # Check numeric values: 2=Yes Wee, 3=Yes Poo, 4=Yes Both
                try:
                    val_int = int(value)
                    if val_int in [2, 3, 4]:
                        is_success = True
                except (ValueError, TypeError):
                    pass
                # Check text values
                if not is_success and value_str.startswith("yes"):
                    is_success = True
            # Also check value_type
            if not is_success and value_type:
                value_type_str = str(value_type).strip().lower()
                if value_type_str.startswith("yes"):
                    is_success = True
            
            if is_success:
                bucket["toilet_success"] += 1
        elif name == "Accidents":
            bucket["accidents"] += 1

        elif name == "Mood":
            num = _safe_float(value) or _safe_float(value_type)
            if num is not None:
                bucket["mood_values"].append(num)

        elif name == "Sleep Quality":
            score = SLEEP_MAP.get(value) or SLEEP_MAP.get(value_type)
            if score:
                bucket["sleep_scores"].append(score)

        elif name == "Communication Level":
            score = get_communication_score(value, value_type)
            if score is not None:
                bucket["communication_scores"].append(score)

        elif name == "Fluid Intake":
            # Check value_type first (contains "50ml", "100ml", etc.), then value (contains codes 1-5)
            ml = _extract_ml(value_type) if value_type else _extract_ml(value)
            bucket["fluid_ml"] += ml

        elif name == "Food":
            sources = []
            if value:
                sources.append(value)
            if value_type:
                sources.extend(part.strip() for part in value_type.split(',') if part.strip())
            for source in sources:
                bucket["food_types"].add(source)
                bucket["food_details"].append(source)  # Store all entries for tooltip

        elif name == "Meltdowns":
            score = MELTDOWN_MAP.get(value) or MELTDOWN_MAP.get(value_type)
            if score is not None:
                bucket["meltdown_scores"].append(score)

        elif name == "Menstrual Cycle":
            score = MENSTRUAL_MAP.get(value) or MENSTRUAL_MAP.get(value_type)
            if score:
                bucket["menstrual_levels"].append(score)

        elif name == "Health / Medical":
            # Extract posture from Health / Medical entries
            if value_type and "posture" in value_type.lower():
                score = _posture_score(value, value_type, notes)
                if score is not None:
                    bucket["posture_scores"].append(score)
            # Count health events and store details
            bucket["health_events"] += 1
            if value_type:
                bucket["health_details"].append(value_type)  # Store full details for tooltip

        elif name == "Walking Ability":
            score = WALKING_MAP.get(value) or WALKING_MAP.get(value_type)
            if score is not None:
                bucket["walking_scores"].append(score)
        
        elif name in ["Fine Motor Skills", "Gross Motor Skills", "Motor Skills"]:
            # Combine all motor skill types into one metric
            score = MOTOR_MAP.get(value) or MOTOR_MAP.get(value_type)
            # Also handle numeric values (1=Poor, 2=Fair, 3=Good)
            if score is None:
                try:
                    val_int = int(value) if value else None
                    if val_int in [1, 2, 3]:
                        score = val_int
                except (ValueError, TypeError):
                    pass
            if score is not None:
                bucket["motor_scores"].append(score)
        
        elif "posture" in name.lower():
            # Legacy: if there's a separate Posture activity
            score = _posture_score(value, value_type, notes)
            if score is not None:
                bucket["posture_scores"].append(score)

        if notes:
            bucket["notes"].append(notes)

    results = []
    for day in sorted(daily.keys()):
        bucket = daily[day]

        mood_avg = round(sum(bucket["mood_values"]) / len(bucket["mood_values"]), 2) if bucket["mood_values"] else None
        sleep_avg = round(sum(bucket["sleep_scores"]) / len(bucket["sleep_scores"]), 2) if bucket["sleep_scores"] else None
        comm_avg = round(sum(bucket["communication_scores"]) / len(bucket["communication_scores"]), 2) if bucket["communication_scores"] else None

        meltdown_max_numeric = max(bucket["meltdown_scores"]) if bucket["meltdown_scores"] else None
        meltdown_label = None
        if meltdown_max_numeric is not None:
            for label, score in MELTDOWN_MAP.items():
                if score == meltdown_max_numeric:
                    meltdown_label = label
                    break

        menstrual_level = max(bucket["menstrual_levels"]) if bucket["menstrual_levels"] else None
        menstrual_label = MENSTRUAL_INVERSE.get(menstrual_level) if menstrual_level else None

        walking_ratio = None
        if bucket["walking_scores"]:
            walking_ratio = round(sum(bucket["walking_scores"]) / len(bucket["walking_scores"]), 2)

        motor_avg = None
        motor_label = None
        if bucket["motor_scores"]:
            motor_avg = round(sum(bucket["motor_scores"]) / len(bucket["motor_scores"]), 2)
            # Map score to label
            if motor_avg >= 2.5:
                motor_label = "Good"
            elif motor_avg >= 1.5:
                motor_label = "Fair"
            else:
                motor_label = "Poor"

        posture_avg = None
        posture_label = None
        if bucket["posture_scores"]:
            posture_avg = round(sum(bucket["posture_scores"]) / len(bucket["posture_scores"]), 2)
            posture_label = _posture_label(posture_avg)

        attempts = bucket["toilet_attempts"]
        successes = bucket["toilet_success"]
        success_rate = round(successes / attempts, 2) if attempts else None

        good_day_score = 0
        if mood_avg is not None and mood_avg >= 4:
            good_day_score += 1
        if success_rate is not None and success_rate >= 0.6:
            good_day_score += 1
        if comm_avg is not None and comm_avg >= 3:
            good_day_score += 1
        if bucket["accidents"] == 0 and attempts:
            good_day_score += 1
        if sleep_avg is not None and sleep_avg >= 3:
            good_day_score += 1

        results.append({
            "date": day.isoformat(),
            "toilet_attempts": attempts,
            "toilet_success": successes,
            "toilet_success_rate": success_rate,
            "accidents": bucket["accidents"],
            "mood_avg": mood_avg,
            "sleep_avg": sleep_avg,
            "communication_avg": comm_avg,
            "meltdown_level": meltdown_label,
            "menstrual_level": menstrual_label,
            "fluid_ml": round(bucket["fluid_ml"], 1),
            "food_types": sorted(bucket["food_types"]),
            "food_count": len(bucket["food_details"]),  # Add food_count for graph
            "health_events": bucket["health_events"],
            "health_summary": format_health_summary(bucket["health_details"]),
            "health_tooltip": format_health_tooltip(bucket["health_details"]),
            "food_summary": format_food_summary(bucket["food_types"], len(bucket["food_details"])),
            "food_tooltip": format_food_tooltip(bucket["food_details"]),
            "walking_independent_ratio": walking_ratio,
            "motor_avg": motor_avg,
            "motor_label": motor_label,
            "good_day_score": good_day_score,
            "notes_highlights": bucket["notes"][:3],
            "posture_avg": posture_avg,
            "posture_label": posture_label,
            "posture_entries": len(bucket["posture_scores"]),
        })

    return results

@app.route('/edit-analysis/<int:analysis_id>', methods=['GET', 'POST'])
def edit_analysis(analysis_id):
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    analysis = AnalysisEntry.query.get_or_404(analysis_id)
    if not can_edit_analysis(analysis):
        abort(403)

    if request.method == 'POST':
        raw_html = request.form.get('analysis_text', '')
        clean_html = bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        if not clean_html.strip():
            flash('Analysis cannot be empty.', 'error')
            return redirect(url_for('edit_analysis', analysis_id=analysis_id))

        analysis.analysis_text = clean_html
        db.session.commit()
        flash('Analysis updated!', 'success')
        return redirect(url_for('view_analyses'))

    return render_template('edit_analysis.html', analysis=analysis)

@app.route('/delete-analysis/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    analysis = AnalysisEntry.query.get_or_404(analysis_id)
    if not can_edit_analysis(analysis):
        return jsonify({'error': 'Unauthorized'}), 403

    db.session.delete(analysis)
    db.session.commit()
    flash('Analysis deleted.', 'success')
    return redirect(url_for('view_analyses'))

db.init_app(app)
bcrypt = Bcrypt(app)

def add_missing_columns():
    with app.app_context():
        inspector = db.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('care_log_entries')]
        if 'value_type' not in columns:
            db.session.execute(text('ALTER TABLE care_log_entries ADD COLUMN value_type VARCHAR(100)'))
            db.session.commit()
            print("Added 'value_type' column to care_log_entries table.")

def analyze_carer_logging(carer_name, log_entries):
    """Analyze a carer's logging patterns and generate insights."""
    from collections import defaultdict
    from datetime import datetime, timedelta
    
    if not log_entries:
        return {
            'activity_frequency': {},
            'quality_metrics': {},
            'consistency_metrics': {},
            'gaps_analysis': {},
            'insights': ['No log entries found for this carer.']
        }
    
    # Activity frequency analysis
    activity_counts = defaultdict(int)
    activity_dates = defaultdict(list)  # Track when each activity was logged
    
    # Quality metrics (for activities with scores)
    mood_scores = []
    comm_scores = []
    sleep_scores = []
    toilet_attempts = 0
    toilet_successes = 0
    meltdown_counts = defaultdict(int)
    
    # Consistency metrics
    log_dates = set()
    log_times = []
    days_with_logs = set()
    
    # All possible activities (for gap analysis)
    all_activities = [
        "Mood", "Sleep Quality", "Communication Level", "Toilet Tries", "Accidents",
        "Food", "Fluid Intake", "Exercise / Physical Activity", "Walking Ability",
        "Motor Skills", "Meltdowns", "Teeth Brushed",
        "Bath/Shower Completed", "Nap Taken", "Sensory Sensitivities", "Menstrual Cycle",
        "Health / Medical", "Message for Careers"
    ]
    
    for entry in log_entries:
        activity_name = entry.activity_name
        # Combine Fine Motor Skills and Gross Motor Skills into Motor Skills
        if activity_name in ["Fine Motor Skills", "Gross Motor Skills"]:
            activity_name = "Motor Skills"
        activity_counts[activity_name] += 1
        log_date = entry.activity_datetime.date()
        log_dates.add(log_date)
        days_with_logs.add(log_date)
        activity_dates[activity_name].append(log_date)
        log_times.append(entry.activity_datetime.hour)
        
        # Quality metrics
        if activity_name == "Mood":
            try:
                mood_scores.append(float(entry.value))
            except (ValueError, TypeError):
                pass
        elif activity_name == "Communication Level":
            score = get_communication_score(entry.value, entry.value_type)
            if score is not None:
                comm_scores.append(score)
        elif activity_name == "Sleep Quality":
            score = SLEEP_MAP.get(entry.value) or SLEEP_MAP.get(entry.value_type)
            if score:
                sleep_scores.append(score)
        elif activity_name == "Toilet Tries":
            toilet_attempts += 1
            # Check both value (numeric) and value_type (text) for success
            value_str = (entry.value or "").lower()
            value_type_str = (entry.value_type or "").lower()
            # Success if value is 2, 3, or 4 (Yes Wee, Yes Poo, Yes Both) OR text starts with "yes"
            if (value_str in ["2", "3", "4"] or 
                value_type_str.startswith("yes") or 
                value_str.startswith("yes")):
                toilet_successes += 1
        elif activity_name == "Meltdowns":
            severity = MELTDOWN_MAP.get(entry.value) or MELTDOWN_MAP.get(entry.value_type)
            if severity is not None:
                meltdown_counts[severity] += 1
    
    # Activity frequency (sorted)
    activity_frequency = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    most_logged = activity_frequency[:5] if len(activity_frequency) >= 5 else activity_frequency
    least_logged = activity_frequency[-5:] if len(activity_frequency) >= 5 else []
    
    # Gaps analysis
    logged_activities = set(activity_counts.keys())
    missing_activities = [act for act in all_activities if act not in logged_activities]
    
    # Days since last log
    if log_dates:
        last_log_date = max(log_dates)
        days_since_last = (date.today() - last_log_date).days
    else:
        days_since_last = None
    
    # Fix: If we have entries but no dates (shouldn't happen, but safety check)
    if days_since_last is None and log_entries:
        # Get last entry date directly
        last_entry = max(log_entries, key=lambda e: e.activity_datetime)
        last_log_date = last_entry.activity_datetime.date()
        days_since_last = (date.today() - last_log_date).days
    
    # Logging frequency
    if log_dates:
        total_days = (max(log_dates) - min(log_dates)).days + 1
        entries_per_day = len(log_entries) / total_days if total_days > 0 else 0
    else:
        entries_per_day = 0
    
    # Most active day of week
    day_counts = defaultdict(int)
    for log_date in days_with_logs:
        day_name = log_date.strftime('%A')
        day_counts[day_name] += 1
    most_active_day = max(day_counts.items(), key=lambda x: x[1]) if day_counts else None
    
    # Most active time of day
    if log_times:
        morning = sum(1 for t in log_times if 6 <= t < 12)
        afternoon = sum(1 for t in log_times if 12 <= t < 18)
        evening = sum(1 for t in log_times if 18 <= t < 24)
        night = sum(1 for t in log_times if 0 <= t < 6)
        time_periods = {
            'Morning (6am-12pm)': morning,
            'Afternoon (12pm-6pm)': afternoon,
            'Evening (6pm-12am)': evening,
            'Night (12am-6am)': night
        }
        most_active_time = max(time_periods.items(), key=lambda x: x[1]) if time_periods else None
    else:
        most_active_time = None
    
    # Activities not logged recently (last 30 days)
    recent_date = date.today() - timedelta(days=30)
    activities_not_recent = []
    for activity, dates in activity_dates.items():
        if dates:  # Only process if we have dates
            recent_dates = [d for d in dates if d >= recent_date]
            if not recent_dates:
                last_logged = max(dates)
                days_ago = (date.today() - last_logged).days
                activities_not_recent.append((activity, days_ago))
    
    # Calculate trends (comparing recent vs older data)
    recent_entries = [e for e in log_entries if e.activity_datetime.date() >= recent_date]
    older_entries = [e for e in log_entries if e.activity_datetime.date() < recent_date]
    
    # Recent vs older mood comparison
    recent_mood = []
    older_mood = []
    for entry in recent_entries:
        if entry.activity_name == "Mood":
            try:
                recent_mood.append(float(entry.value))
            except (ValueError, TypeError):
                pass
    for entry in older_entries:
        if entry.activity_name == "Mood":
            try:
                older_mood.append(float(entry.value))
            except (ValueError, TypeError):
                pass
    
    mood_trend = None
    if recent_mood and older_mood:
        recent_avg = sum(recent_mood) / len(recent_mood)
        older_avg = sum(older_mood) / len(older_mood)
        mood_trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        mood_change = recent_avg - older_avg
    else:
        mood_change = None
    
    # Recent vs older communication comparison
    recent_comm = []
    older_comm = []
    for entry in recent_entries:
        if entry.activity_name == "Communication Level":
            score = get_communication_score(entry.value, entry.value_type)
            if score is not None:
                recent_comm.append(score)
    for entry in older_entries:
        if entry.activity_name == "Communication Level":
            score = get_communication_score(entry.value, entry.value_type)
            if score is not None:
                older_comm.append(score)
    
    comm_trend = None
    if recent_comm and older_comm:
        recent_avg = sum(recent_comm) / len(recent_comm)
        older_avg = sum(older_comm) / len(older_comm)
        comm_trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        comm_change = recent_avg - older_avg
    else:
        comm_change = None
    
    # Recent vs older toilet success rate
    recent_toilet_attempts = 0
    recent_toilet_successes = 0
    older_toilet_attempts = 0
    older_toilet_successes = 0
    
    for entry in recent_entries:
        if entry.activity_name == "Toilet Tries":
            recent_toilet_attempts += 1
            value_str = (entry.value or "").lower()
            value_type_str = (entry.value_type or "").lower()
            if (value_str in ["2", "3", "4"] or value_type_str.startswith("yes") or value_str.startswith("yes")):
                recent_toilet_successes += 1
    
    for entry in older_entries:
        if entry.activity_name == "Toilet Tries":
            older_toilet_attempts += 1
            value_str = (entry.value or "").lower()
            value_type_str = (entry.value_type or "").lower()
            if (value_str in ["2", "3", "4"] or value_type_str.startswith("yes") or value_str.startswith("yes")):
                older_toilet_successes += 1
    
    toilet_trend = None
    if recent_toilet_attempts > 0 and older_toilet_attempts > 0:
        recent_rate = (recent_toilet_successes / recent_toilet_attempts) * 100
        older_rate = (older_toilet_successes / older_toilet_attempts) * 100
        toilet_trend = "improving" if recent_rate > older_rate else "declining" if recent_rate < older_rate else "stable"
        toilet_change = recent_rate - older_rate
    else:
        toilet_change = None
    
    # Generate insights
    insights = []
    
    # Quality insights
    if mood_scores:
        avg_mood = sum(mood_scores) / len(mood_scores)
        if avg_mood >= 4:
            insights.append(f"🌟 Excellent mood tracking! Average mood: {avg_mood:.1f}/5")
        elif avg_mood >= 3:
            insights.append(f"✅ Good mood tracking. Average mood: {avg_mood:.1f}/5")
        else:
            insights.append(f"⚠️ Low mood scores detected. Average: {avg_mood:.1f}/5 - may need attention")
    
    if comm_scores:
        avg_comm = sum(comm_scores) / len(comm_scores)
        if avg_comm >= 3:
            insights.append(f"💬 Strong communication levels! Average: {avg_comm:.1f}/4")
        else:
            insights.append(f"📢 Communication levels average: {avg_comm:.1f}/4")
    
    if sleep_scores:
        avg_sleep = sum(sleep_scores) / len(sleep_scores)
        if avg_sleep >= 3:
            insights.append(f"😴 Good sleep quality tracking. Average: {avg_sleep:.1f}/4")
        else:
            insights.append(f"🌙 Sleep quality average: {avg_sleep:.1f}/4")
    
    # Toilet success rate
    if toilet_attempts > 0:
        success_rate = (toilet_successes / toilet_attempts) * 100
        if success_rate >= 60:
            insights.append(f"🚽 Excellent toilet success rate: {success_rate:.0f}% ({toilet_successes}/{toilet_attempts})")
        else:
            insights.append(f"🚽 Toilet success rate: {success_rate:.0f}% ({toilet_successes}/{toilet_attempts})")
    
    # Consistency insights
    if days_since_last is not None:
        if days_since_last == 0:
            insights.append("✅ Logged today - great consistency!")
        elif days_since_last <= 3:
            insights.append(f"✅ Active logging - last entry {days_since_last} day(s) ago")
        elif days_since_last <= 7:
            insights.append(f"⚠️ Last logged {days_since_last} days ago - consider logging more frequently")
        else:
            insights.append(f"🔴 Last logged {days_since_last} days ago - significant gap")
    
    if entries_per_day > 0:
        if entries_per_day >= 5:
            insights.append(f"📊 High logging frequency: {entries_per_day:.1f} entries per day")
        elif entries_per_day >= 2:
            insights.append(f"📊 Moderate logging frequency: {entries_per_day:.1f} entries per day")
        else:
            insights.append(f"📊 Low logging frequency: {entries_per_day:.1f} entries per day - consider logging more")
    
    # Gap insights
    if activities_not_recent:
        top_gap = max(activities_not_recent, key=lambda x: x[1])
        insights.append(f"⚠️ '{top_gap[0]}' hasn't been logged in {top_gap[1]} days - consider adding this")
    
    if missing_activities:
        insights.append(f"📝 {len(missing_activities)} activity type(s) never logged: {', '.join(missing_activities[:3])}{'...' if len(missing_activities) > 3 else ''}")
    
    # Meltdown insights
    total_meltdowns = sum(meltdown_counts.values())
    if total_meltdowns > 0:
        severe_meltdowns = meltdown_counts.get(3, 0)
        if severe_meltdowns > 0:
            insights.append(f"⚠️ {severe_meltdowns} severe meltdown(s) recorded - monitor closely")
        else:
            insights.append(f"✅ Meltdown tracking: {total_meltdowns} total (mostly mild/moderate)")
    
    # Add trend insights
    if mood_trend and mood_change is not None:
        recent_avg = sum(recent_mood) / len(recent_mood) if recent_mood else 0
        if mood_trend == "improving":
            insights.append(f"📈 Mood is improving! Recent average ({recent_avg:.1f}) is {mood_change:+.1f} higher than previous period")
        elif mood_trend == "declining":
            insights.append(f"📉 Mood trend declining. Recent average ({recent_avg:.1f}) is {mood_change:+.1f} lower - monitor closely")
    
    if comm_trend and comm_change is not None:
        recent_avg = sum(recent_comm) / len(recent_comm) if recent_comm else 0
        if comm_trend == "improving":
            insights.append(f"📈 Communication improving! Recent average ({recent_avg:.1f}) is {comm_change:+.1f} higher")
        elif comm_trend == "declining":
            insights.append(f"📉 Communication declining. Recent average ({recent_avg:.1f}) is {comm_change:+.1f} lower")
    
    if toilet_trend and toilet_change is not None and recent_toilet_attempts > 0:
        recent_rate = (recent_toilet_successes / recent_toilet_attempts * 100)
        if toilet_trend == "improving":
            insights.append(f"📈 Toilet success improving! Recent rate ({recent_rate:.0f}%) is {toilet_change:+.0f}% higher than previous")
        elif toilet_trend == "declining":
            insights.append(f"📉 Toilet success declining. Recent rate ({recent_rate:.0f}%) is {toilet_change:+.0f}% lower")
    
    return {
        'activity_frequency': {
            'most_logged': most_logged,
            'least_logged': least_logged,
            'total_unique': len(logged_activities)
        },
        'quality_metrics': {
            'avg_mood': sum(mood_scores) / len(mood_scores) if mood_scores else None,
            'avg_communication': sum(comm_scores) / len(comm_scores) if comm_scores else None,
            'avg_sleep': sum(sleep_scores) / len(sleep_scores) if sleep_scores else None,
            'toilet_success_rate': (toilet_successes / toilet_attempts * 100) if toilet_attempts > 0 else None,
            'total_meltdowns': total_meltdowns,
            'severe_meltdowns': meltdown_counts.get(3, 0),
            'toilet_attempts': toilet_attempts,
            'toilet_successes': toilet_successes
        },
        'consistency_metrics': {
            'days_since_last_log': days_since_last,
            'entries_per_day': entries_per_day,
            'total_days_logged': len(days_with_logs),
            'most_active_day': most_active_day,
            'most_active_time': most_active_time
        },
        'gaps_analysis': {
            'missing_activities': missing_activities,
            'activities_not_recent': sorted(activities_not_recent, key=lambda x: x[1], reverse=True)[:5],
            'total_missing': len(missing_activities)
        },
        'trends': {
            'mood_trend': mood_trend,
            'mood_change': mood_change,
            'comm_trend': comm_trend,
            'comm_change': comm_change,
            'toilet_trend': toilet_trend,
            'toilet_change': toilet_change
        },
        'insights': insights
    }

def compare_carer_to_others(this_carer_name, this_carer_logs, all_logs):
    """Compare this carer's logging patterns against other carers to find gaps."""
    from datetime import date, timedelta
    
    if not this_carer_logs:
        return {
            'missing_activities': [],
            'recent_missing': [],
            'comparison_period': None
        }
    
    # For recent period (last 30 days), what did others log that this carer didn't?
    recent_date = date.today() - timedelta(days=30)
    this_carer_recent = [
        e for e in this_carer_logs 
        if e.activity_datetime.date() >= recent_date
    ]
    other_carers_recent = [
        e for e in all_logs 
        if e.carer_name != this_carer_name 
        and e.activity_datetime.date() >= recent_date
    ]
    
    this_carer_recent_activities = set(e.activity_name for e in this_carer_recent)
    other_carers_recent_activities = set(e.activity_name for e in other_carers_recent)
    
    recent_missing = []
    for activity in other_carers_recent_activities - this_carer_recent_activities:
        # Find when other carers logged this activity recently
        other_logs = [e for e in other_carers_recent if e.activity_name == activity]
        if other_logs:
            most_recent = max(other_logs, key=lambda x: x.activity_datetime)
            days_ago = (date.today() - most_recent.activity_datetime.date()).days
            # Count how many other carers logged this
            carers_who_logged = set(e.carer_name for e in other_logs)
            
            # Get detailed log entries (most recent 5 per carer)
            detailed_logs = []
            for carer in carers_who_logged:
                carer_logs = [e for e in other_logs if e.carer_name == carer]
                carer_logs.sort(key=lambda x: x.activity_datetime, reverse=True)
                for log in carer_logs[:5]:  # Max 5 most recent per carer
                    detailed_logs.append({
                        'carer': carer,
                        'date': log.activity_datetime.date(),
                        'datetime': log.activity_datetime,
                        'value': log.value,
                        'value_type': log.value_type,
                        'notes': log.notes[:100] if log.notes else None,  # Truncate long notes
                        'duration': log.duration
                    })
            
            # Sort detailed logs by date (most recent first)
            detailed_logs.sort(key=lambda x: x['datetime'], reverse=True)
            
            recent_missing.append({
                'activity': activity,
                'days_ago': days_ago,
                'last_logged_date': most_recent.activity_datetime.date(),
                'logged_by': list(carers_who_logged),
                'count': len(other_logs),
                'detailed_logs': detailed_logs[:10]  # Max 10 most recent entries total
            })
    
    # Sort by most recent first
    recent_missing.sort(key=lambda x: x['days_ago'])
    
    return {
        'missing_activities': [],
        'recent_missing': recent_missing,
        'comparison_period': None
    }

def add_user_profile_columns():
    """Add profile columns to users table if they don't exist."""
    with app.app_context():
        inspector = db.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('users')]
        
        profile_fields = {
            'full_name': 'VARCHAR(200)',
            'phone': 'VARCHAR(50)',
            'education': 'TEXT',
            'qualifications': 'TEXT',
            'skills': 'TEXT',
            'strengths': 'TEXT',
            'experience_years': 'INTEGER',
            'specializations': 'TEXT',
            'bio': 'TEXT',
            'hire_date': 'DATE',
            'notes': 'TEXT'
        }
        
        for field_name, field_type in profile_fields.items():
            if field_name not in columns:
                try:
                    db.session.execute(text(f'ALTER TABLE users ADD COLUMN {field_name} {field_type}'))
                    db.session.commit()
                    print(f"Added '{field_name}' column to users table.")
                except Exception as e:
                    print(f"Error adding {field_name}: {e}")
                    db.session.rollback()

def seed_gas_goals():
    """Insert baseline GAS goals for Tamara if not present."""
    from models import GASGoal
    existing = GASGoal.query.filter_by(person_name='Tamara Fitzmaurice').count()
    if existing:
        return

    goals = [
        dict(
            goal_no=1,
            competency="Linguistic/Social",
            short_label="Requests single words",
            baseline_text="Tamara currently looks at the item she wants and the communication partner most of the time.",
            minus1_text="Tamara will use single words on their device to ask for desired items when provided with a gestural cue (e.g. tapping on device), by the end of a 4-week trial.",
            zero_text="Tamara will use single words on their device to ask for desired items when their communication partner makes a comment as a prompt (e.g. “it looks like you want something”), by the end of a 4-week trial.",
            plus1_text="Tamara will use single words on their device to ask for desired items when provided with an expectant/thoughtful pause, by the end of a 4-week trial.",
            plus2_text="Tamara will use single words on their device to ask for a desired item, independently, by the end of a 4-week trial.",
        ),
        dict(
            goal_no=2,
            competency="Operational",
            short_label="2-hit selection + navigate back",
            baseline_text="Tamara sometimes (inconsistently) attempts to touch the AAC system when prompted.",
            minus1_text="Tamara will consistently touch a button on the screen once (make a 1-hit selection) to select a word when given a visual/verbal prompt, by the end of a 4-week trial.",
            zero_text="Tamara will touch two buttons (make a 2-hit selection by navigating through a folder) to select a word when given a visual prompt (e.g. pointing), by the end of a 4-week trial.",
            plus1_text="Tamara will touch two buttons (make a 2-hit selection) to select a word, independently, by the end of a 4-week trial.",
            plus2_text="Tamara will make a 2-hit selection to select a word and navigate back to home (back/core button) when given a visual prompt, by the end of a 4-week trial.",
        ),
        dict(
            goal_no=3,
            competency="Strategic/Social",
            short_label="Gain partner’s attention with AAC",
            baseline_text="Tamara will gain partner’s attention by looking at them and/or looking at the item she wants.",
            minus1_text="Tamara will gain partner’s attention by coming up to communication partner and looking/touching/vocalising/gesturing.",
            zero_text="Tamara will gain partner’s attention by reaching out to the AAC device and looking at communication partner.",
            plus1_text="Tamara will gain partner’s attention by expressing symbols on the AAC device and looking at communication partner.",
            plus2_text="Tamara will gain partner’s attention by picking up the AAC device and bringing it to partner (may also vocalise or touch).",
        ),
    ]
    for g in goals:
        db.session.add(GASGoal(person_name='Tamara Fitzmaurice', **g))
    db.session.commit()
    print("Seeded GAS goals for Tamara.")


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'heic'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with app.app_context():
    db.create_all()
    add_missing_columns()
    add_user_profile_columns()

@app.context_processor
def utility_processor():
    return dict(User=User)

@app.route('/')
def index():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('log_entry'))

@app.route('/home')
def home_page():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/log', methods=['GET', 'POST'])
def log_entry():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            duration = request.form.get('duration')
            activity_name = request.form['activity_name']
            calculated_value = request.form.get('calculated_value')

            # Core value + value_type handling (unchanged)
            if activity_name in ["Exercise / Physical Activity", "Food"]:
                values = request.form.getlist('value')
                value = calculated_value if calculated_value else str(len(values))
                value_type = request.form.get('value_type', ','.join(values))
            else:
                value = request.form['value']
                value_type = request.form.get('value_type', '')

            # Parse the activity timestamp once so sidecar records can reuse it
            activity_dt = datetime.strptime(request.form['activity_datetime'], '%Y-%m-%dT%H:%M')

            # Main CareLogEntry
            log = CareLogEntry(
                carer_id=session['carer_id'],
                carer_name=session['carer_name'],
                activity_name=activity_name,
                value=value,
                value_type=value_type,
                notes=request.form['notes'],
                activity_datetime=activity_dt,
                duration=duration,
                activity_type='manual'
            )
            db.session.add(log)
            db.session.commit()

            # Save media (unchanged)
            if 'media' in request.files:
                files = request.files.getlist('media')
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        media_path = os.path.join('uploads_test', filename).replace(os.sep, '/')
                        media = Media(path=media_path, entry_id=log.id)
                        db.session.add(media)
                        db.session.commit()

            # -----------------------------
            # AAC / GAS sidecar save logic
            # -----------------------------
            try:
                if activity_name == "Communication Level":
                    aac_prompt = (request.form.get('aac_prompting_level') or '').strip()
                    aac_words  = (request.form.get('aac_words_used') or '').strip()
                    ipad_use = (request.form.get('ipad_use') or '').strip()
                    aac_level_raw = request.form.get('aac_attainment_level')
                    also_aac  = request.form.get('also_create_aac') == 'on'
                    also_prompt_log = request.form.get('also_make_prompt_log') == 'on'

                    # tidy attainment level (optional)
                    aac_level = None
                    if aac_level_raw not in (None, '',):
                        try:
                            aac_level = int(aac_level_raw)
                        except ValueError:
                            aac_level = None

                    # 2a) Create AACTrialEntry if requested and we have any signal
                    if also_aac and (aac_prompt or aac_level is not None or aac_words):
                        from models import AACTrialEntry  # local import to avoid top-level reorder
                        observation_clean = bleach.clean(
                            request.form.get('notes') or '',
                            tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True
                        )
                        entry = AACTrialEntry(
                            person_name='Tamara Fitzmaurice',
                            goal_no=1,  # adjust if you want to map to a particular goal
                            attainment_level=aac_level if aac_level is not None else 0,
                            date=activity_dt.date(),
                            time_of_day=activity_dt.strftime('%H:%M'),
                            location='',
                            partners='',
                            device='',
                            vocabulary='',
                            trial_window='',
                            prompting_level=aac_prompt,
                            modeled_words='',
                            used_words=aac_words,
                            observation=observation_clean,
                            comments='Auto-created from Communication Level log',
                            recorded_by=session.get('carer_name')
                        )
                        db.session.add(entry)

                    # 2b) Also add a classic log row summarising the prompting level
                    if also_prompt_log and (aac_prompt or aac_level is not None or aac_words or ipad_use):
                        prompt_notes_bits = []
                        if ipad_use:
                            prompt_notes_bits.append(f"iPad use: {ipad_use}")
                        if aac_words:
                            prompt_notes_bits.append(f"words used: {aac_words}")
                        if aac_level is not None:
                            prompt_notes_bits.append(f"attainment level: {aac_level}")
                        prompt_notes = '; '.join(prompt_notes_bits)

                        prompt_log = CareLogEntry(
                            carer_id=session['carer_id'],
                            carer_name=session['carer_name'],
                            activity_name='GAS & AAC Prompting Level',
                            value=str(aac_level if aac_level is not None else 0),
                            value_type=aac_prompt or '',
                            notes=prompt_notes,
                            activity_datetime=activity_dt,
                            duration=duration,
                            activity_type='manual'
                        )
                        db.session.add(prompt_log)

                    # Commit sidecars if any
                    db.session.commit()
            except Exception as sidecar_err:
                db.session.rollback()
                # Don't break the main log — just record the failure
                print(f"AAC/GAS sidecar save failed: {sidecar_err}")

            # Sheets sync (unchanged, uses the main `log`)
            sync_to_google_sheets(log)
            return '', 200

        except RequestEntityTooLarge:
            return jsonify({'error': 'Upload size exceeds 50MB. Please remove some files and try again.'}), 413
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Error logging entry: {str(e)}'}), 500

    # GET
    now = datetime.now().strftime('%Y-%m-%dT%H:%M')
    return render_template('log_entry.html', current_datetime=now)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password) and user.is_approved:
            session['carer_id'] = username
            session['carer_name'] = username
            session['just_logged_in'] = True  # Flag to show notice modal
            flash('Login successful!', 'success')
            # Check for unread notices
            unread_count = get_unread_notice_count()
            if unread_count > 0:
                flash(f'You have {unread_count} unread notice{"s" if unread_count > 1 else ""}! Check the Notice Board.', 'info')
            return redirect(url_for('log_entry'))
        flash('Invalid credentials or account not approved!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists!', 'error')
            return redirect(url_for('register'))
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=password_hash, is_approved=False)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Awaiting admin approval.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/clear-login-flag', methods=['POST'])
def clear_login_flag():
    """Clear the just_logged_in session flag after modal is shown"""
    if 'carer_id' in session:
        session.pop('just_logged_in', None)
    return jsonify({'success': True})

@app.route('/view-logs', methods=['GET'])
def view_logs():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    # Get filter parameters
    activity = request.args.get('activity')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    carer_name = request.args.get('carer_name')

    # Build query
    query = CareLogEntry.query
    if activity:
        # Combine Fine Motor Skills and Gross Motor Skills when filtering by Motor Skills
        if activity == "Motor Skills":
            query = query.filter(
                CareLogEntry.activity_name.in_(["Fine Motor Skills", "Gross Motor Skills", "Motor Skills"])
            )
        else:
            query = query.filter(CareLogEntry.activity_name == activity)
    if start_date:
        query = query.filter(CareLogEntry.activity_datetime >= start_date)
    if end_date:
        query = query.filter(CareLogEntry.activity_datetime <= end_date)
    if carer_name:
        query = query.filter(CareLogEntry.carer_name == carer_name)

    logs = query.order_by(CareLogEntry.activity_datetime).all()

    # Get unique carer names for filter dropdown
    carer_names = db.session.query(CareLogEntry.carer_name).distinct().all()
    carer_names = [name[0] for name in carer_names]

    non_numeric_activities = ["Menstrual Cycle"]
    numeric_logs = []
    for l in logs:
        if l.activity_name in non_numeric_activities:
            continue
        try:
            float(l.value)
            numeric_logs.append(l)
        except (ValueError, TypeError):
            continue

    labels = sorted(set(l.activity_datetime.strftime('%Y-%m-%d %H:%M') for l in numeric_logs))

    from collections import defaultdict

    activity_data = defaultdict(lambda: {label: None for label in labels})
    tooltip_data = defaultdict(lambda: {label: {'value_type': '', 'notes': '', 'duration': ''} for label in labels})

    for l in numeric_logs:
        label = l.activity_datetime.strftime('%Y-%m-%d %H:%M')
        # Combine Fine Motor Skills and Gross Motor Skills into Motor Skills
        activity_name = l.activity_name
        if activity_name in ["Fine Motor Skills", "Gross Motor Skills"]:
            activity_name = "Motor Skills"
        activity_data[activity_name][label] = float(l.value)
        tooltip_data[activity_name][label] = {
            'value_type': l.value_type or '',
            'notes': l.notes or '',
            'duration': l.duration or ''
        }

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'teal', 'brown', 'gray']

    event_based = ["Toilet Tries", "Accidents", "Meltdowns", "Teeth Brushed", "Bath/Shower Completed", "Nap Taken", "Food", "Sensory Sensitivities", "Message for Careers", "Menstrual Cycle", "Health / Medical"]
    line_based = ["Mood", "Sleep Quality", "Communication Level", "Walking Ability", "Motor Skills", "Fluid Intake", "Exercise / Physical Activity"]

    datasets = []
    for i, (activity_name, values_by_date) in enumerate(activity_data.items()):
        is_event = activity_name in event_based
        datasets.append({
            "type": "bar" if is_event else "line",
            "label": activity_name,
            "data": [
                {
                    'x': label,
                    'y': values_by_date[label] if is_event else values_by_date[label],
                    'value_type': tooltip_data[activity_name][label]['value_type'],
                    'notes': tooltip_data[activity_name][label]['notes'],
                    'duration': tooltip_data[activity_name][label]['duration']
                }
                for label in labels
            ],
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] if is_event else 'rgba(0,0,0,0)',
            "fill": False,
            "tension": 0.3 if not is_event else 0
        })

    for log in logs:
        log.display_value = log.value_type if log.value_type else log.value

    combined_data_by_time = defaultdict(list)
    for activity_name, values_by_date in activity_data.items():
        for label, val in values_by_date.items():
            if val is not None:
                combined_data_by_time[label].append(val)

    combined_data = []
    for label in labels:
        values = combined_data_by_time[label]
        avg = sum(values) / len(values) if values else None
        combined_data.append({"x": label, "y": avg})

    combined_datasets = [{
        "label": "Combined Activity Average",
        "data": combined_data,
        "borderColor": "black",
        "backgroundColor": "rgba(0,0,0,0)",
        "fill": False,
        "tension": 0.2
    }]

    return render_template(
        'view_logs.html',
        logs=logs,
        labels=labels,
        datasets=datasets,
        combined_datasets=combined_datasets,
        carer_names=carer_names,
        selected_carer=carer_name
    )

@app.route('/admin')
def admin():
    if 'carer_id' not in session or not User.query.filter_by(username=session['carer_id']).first().is_admin:
        return redirect(url_for('login'))
    users = User.query.all()
    # Get unique carer names from log entries
    carer_names = db.session.query(CareLogEntry.carer_name).distinct().all()
    carer_names = [name[0] for name in carer_names if name[0]]
    return render_template('admin.html', users=users, carer_names=sorted(carer_names))

@app.route('/admin/approve/<int:user_id>', methods=['POST'])
def approve_user(user_id):
    if 'carer_id' not in session or not User.query.filter_by(username=session['carer_id']).first().is_admin:
        return redirect(url_for('login'))
    user = User.query.get_or_404(user_id)
    if not user.is_approved:
        user.is_approved = True
        db.session.commit()
        flash('User approved!', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
def toggle_admin(user_id):
    if 'carer_id' not in session or not User.query.filter_by(username=session['carer_id']).first().is_admin:
        return redirect(url_for('login'))
    user = User.query.get_or_404(user_id)
    current_user = User.query.filter_by(username=session['carer_id']).first()
    if user.username != current_user.username:
        user.is_admin = not user.is_admin
        db.session.commit()
        flash(f'User {user.username} admin status toggled to {"on" if user.is_admin else "off"}!', 'success')
    else:
        flash('You cannot toggle your own admin status!', 'warning')
    return redirect(url_for('admin'))

@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'carer_id' not in session or not User.query.filter_by(username=session['carer_id']).first().is_admin:
        return redirect(url_for('login'))
    user = User.query.get_or_404(user_id)
    if user.username != session['carer_id']:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted!', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/carer-profile/<int:user_id>', methods=['GET', 'POST'])
def carer_profile(user_id):
    # Check authentication - prevent redirect loops
    if 'carer_id' not in session:
        # Check if it's a bot/crawler (like WhatsApp link preview)
        user_agent = request.headers.get('User-Agent', '').lower()
        is_bot = any(bot in user_agent for bot in ['bot', 'crawler', 'spider', 'whatsapp', 'facebook', 'telegram'])
        
        if is_bot:
            # Return 401 for bots instead of redirect to prevent loops
            return 'Unauthorized - Please log in to view carer profiles', 401
        
        return redirect(url_for('login'))
    
    # Check if user is admin
    current_user = User.query.filter_by(username=session['carer_id']).first()
    if not current_user or not current_user.is_admin:
        return redirect(url_for('login'))
    
    # Handle carer selection from dropdown
    selected_carer_id = request.args.get('carer_select')
    if selected_carer_id:
        try:
            user_id = int(selected_carer_id)
        except (ValueError, TypeError):
            pass  # Use original user_id if invalid
    
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        try:
            # Update profile fields
            user.full_name = request.form.get('full_name', '').strip() or None
            user.phone = request.form.get('phone', '').strip() or None
            user.education = request.form.get('education', '').strip() or None
            user.qualifications = request.form.get('qualifications', '').strip() or None
            user.skills = request.form.get('skills', '').strip() or None
            user.strengths = request.form.get('strengths', '').strip() or None
            user.specializations = request.form.get('specializations', '').strip() or None
            user.bio = request.form.get('bio', '').strip() or None
            user.notes = request.form.get('notes', '').strip() or None
            
            # Handle experience_years
            exp_years = request.form.get('experience_years', '').strip()
            user.experience_years = int(exp_years) if exp_years and exp_years.isdigit() else None
            
            # Handle hire_date
            hire_date_str = request.form.get('hire_date', '').strip()
            if hire_date_str:
                try:
                    user.hire_date = datetime.strptime(hire_date_str, '%Y-%m-%d').date()
                except ValueError:
                    user.hire_date = None
            else:
                user.hire_date = None
            
            db.session.commit()
            flash('Carer profile updated successfully!', 'success')
            return redirect(url_for('carer_profile', user_id=user_id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'error')
    
    # Get statistics for this carer
    carer_name = user.username
    
    # Get filter parameters
    activity_filter = request.args.get('activity')
    start_date_filter = request.args.get('start_date')
    end_date_filter = request.args.get('end_date')
    
    # Build query with filters for display
    query = CareLogEntry.query.filter_by(carer_name=carer_name)
    
    if activity_filter:
        query = query.filter(CareLogEntry.activity_name == activity_filter)
    if start_date_filter:
        query = query.filter(CareLogEntry.activity_datetime >= start_date_filter)
    if end_date_filter:
        query = query.filter(CareLogEntry.activity_datetime <= end_date_filter)
    
    # Get filtered log entries (most recent first, limit to 100 for display)
    filtered_log_entries = query.order_by(CareLogEntry.activity_datetime.desc()).limit(100).all()
    
    # Get all entries for statistics (unfiltered)
    all_log_entries = CareLogEntry.query.filter_by(carer_name=carer_name).all()
    analysis_entries = AnalysisEntry.query.filter_by(carer_name=carer_name).all()
    aac_entries = AACTrialEntry.query.filter_by(recorded_by=carer_name).all()
    
    # Activity breakdown (from all entries for accurate stats)
    from collections import defaultdict
    activity_counts = defaultdict(int)
    for entry in all_log_entries:
        activity_counts[entry.activity_name] += 1
    
    # Get unique activity names for filter dropdown
    unique_activities = db.session.query(CareLogEntry.activity_name).filter_by(carer_name=carer_name).distinct().all()
    unique_activities = sorted([act[0] for act in unique_activities])
    
    # Convert to sorted list for template (Jinja2 doesn't have list() function)
    activity_counts_sorted = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Generate analytics
    analytics = analyze_carer_logging(carer_name, all_log_entries)
    
    # NEW: Compare this carer to others
    all_carer_logs = CareLogEntry.query.all()  # Get ALL logs from ALL carers
    comparison = compare_carer_to_others(carer_name, all_log_entries, all_carer_logs)
    
    # Add comparison data to analytics
    analytics['comparison'] = comparison
    
    # Get all users for dropdown
    all_users = User.query.order_by(User.username).all()
    
    return render_template(
        'carer_profile.html',
        user=user,
        all_users=all_users,  # For dropdown
        total_logs=len(all_log_entries),
        total_analyses=len(analysis_entries),
        total_aac_entries=len(aac_entries),
        activity_counts=activity_counts_sorted,
        log_entries=filtered_log_entries,  # Filtered entries for display
        unique_activities=unique_activities,  # For dropdown
        selected_activity=activity_filter or '',
        selected_start_date=start_date_filter or '',
        selected_end_date=end_date_filter or '',
        analytics=analytics  # Analytics data
    )

@app.route('/admin/all-carers-analytics')
def all_carers_analytics():
    """Display analytics for all carers in a comparison table."""
    if 'carer_id' not in session or not User.query.filter_by(username=session['carer_id']).first().is_admin:
        return redirect(url_for('login'))
    
    # Get filter parameters
    sort_by = request.args.get('sort_by', 'total_entries')
    sort_order = request.args.get('sort_order', 'desc')
    date_filter = request.args.get('date_filter', '30')  # days
    min_entries = request.args.get('min_entries', '0')
    
    try:
        date_filter_days = int(date_filter)
    except (ValueError, TypeError):
        date_filter_days = 30
    
    try:
        min_entries_int = int(min_entries)
    except (ValueError, TypeError):
        min_entries_int = 0
    
    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=date_filter_days)
    
    # Get all approved users
    all_users = User.query.filter_by(is_approved=True).all()
    
    # Get all log entries in date range
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    all_logs = CareLogEntry.query.filter(
        CareLogEntry.activity_datetime >= start_dt,
        CareLogEntry.activity_datetime <= end_dt
    ).all()
    
    # Aggregate analytics for each carer
    carer_analytics = []
    for user in all_users:
        carer_logs = [log for log in all_logs if log.carer_name == user.username]
        
        if len(carer_logs) < min_entries_int:
            continue
        
        analytics = analyze_carer_logging(user.username, carer_logs)
        
        # Get last log date
        last_log_date = None
        if carer_logs:
            last_log_date = max([log.activity_datetime.date() for log in carer_logs])
        
        # Get activity frequency (most logged activities)
        activity_frequency = analytics.get('activity_frequency', {})
        most_logged = activity_frequency.get('most_logged', [])[:3] if activity_frequency.get('most_logged') else []
        
        carer_analytics.append({
            'user': user,
            'username': user.username,
            'full_name': user.full_name or user.username,
            'total_entries': len(carer_logs),
            'days_since_last_log': analytics['consistency_metrics'].get('days_since_last_log'),
            'entries_per_day': analytics['consistency_metrics'].get('entries_per_day', 0),
            'avg_mood': analytics['quality_metrics'].get('avg_mood'),
            'avg_communication': analytics['quality_metrics'].get('avg_communication'),
            'avg_sleep': analytics['quality_metrics'].get('avg_sleep'),
            'toilet_success_rate': analytics['quality_metrics'].get('toilet_success_rate'),
            'activity_coverage': activity_frequency.get('total_unique', 0),
            'most_active_day': analytics['consistency_metrics'].get('most_active_day'),
            'last_log_date': last_log_date,
            'most_logged_activities': most_logged,
            'analytics': analytics
        })
    
    # Sort the results
    reverse_order = (sort_order == 'desc')
    if sort_by == 'username':
        carer_analytics.sort(key=lambda x: x['username'].lower(), reverse=reverse_order)
    elif sort_by == 'total_entries':
        carer_analytics.sort(key=lambda x: x['total_entries'], reverse=reverse_order)
    elif sort_by == 'days_since_last_log':
        carer_analytics.sort(key=lambda x: x['days_since_last_log'] if x['days_since_last_log'] is not None else 999, reverse=not reverse_order)
    elif sort_by == 'entries_per_day':
        carer_analytics.sort(key=lambda x: x['entries_per_day'], reverse=reverse_order)
    elif sort_by == 'avg_mood':
        carer_analytics.sort(key=lambda x: x['avg_mood'] if x['avg_mood'] is not None else 0, reverse=reverse_order)
    elif sort_by == 'activity_coverage':
        carer_analytics.sort(key=lambda x: x['activity_coverage'], reverse=reverse_order)
    
    # Calculate overall averages
    total_entries_all = sum(c['total_entries'] for c in carer_analytics)
    avg_entries_per_carer = total_entries_all / len(carer_analytics) if carer_analytics else 0
    
    # Calculate average mood across all carers
    moods = [c['avg_mood'] for c in carer_analytics if c['avg_mood'] is not None]
    avg_mood_all = sum(moods) / len(moods) if moods else None
    
    # Calculate average communication
    comms = [c['avg_communication'] for c in carer_analytics if c['avg_communication'] is not None]
    avg_comm_all = sum(comms) / len(comms) if comms else None
    
    # Calculate average toilet success rate
    toilet_rates = [c['toilet_success_rate'] for c in carer_analytics if c['toilet_success_rate'] is not None]
    avg_toilet_all = sum(toilet_rates) / len(toilet_rates) if toilet_rates else None
    
    return render_template('all_carers_analytics.html',
                         carer_analytics=carer_analytics,
                         sort_by=sort_by,
                         sort_order=sort_order,
                         date_filter=date_filter,
                         min_entries=min_entries,
                         total_carers=len(carer_analytics),
                         total_entries_all=total_entries_all,
                         avg_entries_per_carer=avg_entries_per_carer,
                         avg_mood_all=avg_mood_all,
                         avg_comm_all=avg_comm_all,
                         avg_toilet_all=avg_toilet_all)

@app.route('/edit-log/<int:log_id>', methods=['GET', 'POST'])
def edit_log(log_id):
    log = CareLogEntry.query.get_or_404(log_id)

    if request.method == 'POST':
        try:
            log.activity_name = request.form['activity_name']
            log.activity_type = request.form.get('activity_type')
            if log.activity_name in ["Exercise / Physical Activity", "Food"]:
                values = request.form.getlist('value')
                log.value = request.form.get('calculated_value', str(len(values)))
                log.value_type = request.form.get('value_type', ','.join(values))
            else:
                log.value = request.form['value']
                log.value_type = request.form.get('value_type', '')
            log.notes = request.form['notes']
            log.duration = request.form.get('duration')
            log.activity_datetime = datetime.strptime(request.form['activity_datetime'], '%Y-%m-%dT%H:%M')

            if 'media' in request.files:
                files = request.files.getlist('media')
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        media_path = os.path.join('uploads_test', filename).replace(os.sep, '/')
                        media = Media(path=media_path, entry_id=log.id)
                        db.session.add(media)

            db.session.commit()
            return '', 200
        except RequestEntityTooLarge:
            return jsonify({'error': 'Upload size exceeds 50MB. Please remove some files and try again.'}), 413
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Error updating entry: {str(e)}'}), 500

    activityValues = {
        "Mood": ["1", "2", "3", "4", "5"],
        "Sleep Quality": ["Poor", "Fair", "Good", "Excellent"],
        "Nap Taken": ["Yes", "No"],
        "Toilet Tries": ["Yes Both", "Yes Poo", "Yes Wee", "No"],
        "Accidents": ["Wet", "Soiled", "Both"],
        "Bath/Shower Completed": ["Yes", "No"],
        "Teeth Brushed": ["Yes", "No"],
        "Walking Ability": ["Needs Help", "Independent"],
        "Motor Skills": ["Poor", "Fair", "Good"],
        "Communication Level": ["No communication", "Passive communication", "Minimal communication, not directed at others", "Directed communication without back-and-forth", "Back and forth communication"],
        "Fluid Intake": ["50ml", "100ml", "200ml", "300ml", "500ml+"],
        "Meltdowns": ["None", "Mild", "Moderate", "Severe"],
        "Sensory Sensitivities": ["Sound", "Light", "Touch", "Movement"],
        "Exercise / Physical Activity": ["1.5", "2.0", "2.5", "3.5", "4.2", "4.0", "4.5", "5.0"],
        "Menstrual Cycle": ["Spotting", "Medium", "Heavy"],
        "Food": ["Protein", "Vegetables", "Carbohydrate", "Fruit", "Cereals", "Other"],
        "Message for Careers": ["Message"],
        "Health / Medical": [
            "Medication Taken",
            "Doctor's Appointment",
            "Hospital Visit",
            "Large Bloating",
            "Less Bloated",
            "Raynauds Purple Hands or Feet",
            "Raynauds Red Hands or Feet",
            "Dribbling allot",
            "Dribbling less",
            "Other"
        ]
    }

    return render_template("edit_log.html", log=log, activityValues=activityValues)

@app.route('/delete-log/<int:log_id>', methods=['POST'])
def delete_log(log_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(username=session['carer_id']).first()
    log = CareLogEntry.query.get_or_404(log_id)
    
    # Restrict deletion to the log's creator or admins
    if log.carer_id != session['carer_id'] and not user.is_admin:
        return jsonify({'error': 'Unauthorized: You can only delete your own logs or must be an admin'}), 403

    try:
        # Delete associated media files
        for media in log.media:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(media.path))
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete the log (cascades to media due to cascade="all, delete-orphan")
        db.session.delete(log)
        db.session.commit()
        
        # Skip Google Sheets sync for deletion until implemented
        print("Skipping Google Sheets sync for deletion (not implemented)")
        
        return '', 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting log: {str(e)}'}), 500

@app.route('/delete-media/<int:media_id>', methods=['POST'])
def delete_media(media_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    media = Media.query.get_or_404(media_id)
    log = CareLogEntry.query.get_or_404(media.entry_id)
    if log.carer_id != session['carer_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(media.path))
        if os.path.exists(file_path):
            os.remove(file_path)
        db.session.delete(media)
        db.session.commit()
        return '', 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error deleting media: {str(e)}'}), 500

@app.route('/view-media/<int:log_id>')
def view_media(log_id):
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    log = CareLogEntry.query.get_or_404(log_id)
    media_list = log.media
    return render_template('view_media.html', media_list=media_list, log=log)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Accepts both 'image' (View Media page) and 'media' (View Logs page)
    file = request.files.get('media') or request.files.get('image')
    entry_id = request.form.get('entry_id') or request.form.get('log_id')

    if not file or not entry_id:
        return jsonify({'success': False, 'error': 'Missing file or entry ID'}), 400

    # Use the configured uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']  # static/uploads_test
    os.makedirs(upload_folder, exist_ok=True)

    # Generate safe and unique filename
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Save DB record (store relative path used by templates)
    relative_path = os.path.join('uploads_test', filename).replace(os.sep, '/')
    new_media = Media(entry_id=entry_id, path=relative_path)
    db.session.add(new_media)
    db.session.commit()

    return jsonify({'success': True, 'filename': filename})


from models import GASGoal, AACTrialEntry



@app.route('/aac', methods=['GET'])
def aac_tracker_home():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    goals = GASGoal.query.filter_by(person_name='Tamara Fitzmaurice').order_by(GASGoal.goal_no.asc()).all()
    entries = (AACTrialEntry.query.filter_by(person_name='Tamara Fitzmaurice')
               .order_by(AACTrialEntry.created_at.desc()).limit(20).all())

    # Pass full datetime string for <input type="datetime-local">
    return render_template(
        'aac_tracker.html',
        goals=goals,
        entries=entries,
        PROMPTING_LEVELS=PROMPTING_LEVELS,
        now=datetime.now().strftime("%Y-%m-%dT%H:%M")   # ✅ e.g., 2025-10-06T14:30
    )


@app.route('/aac/save', methods=['POST'])
def aac_save_entry():
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        goal_no = int(request.form.get('goal_no', '0') or 0)
        attainment_level = int(request.form.get('attainment_level', '0') or 0)

        # Diary/Context
        # Expecting YYYY-MM-DDTHH:MM from <input type="datetime-local" name="date">
        dt_str = request.form.get('date', '')  # may be '', 'YYYY-MM-DD', or 'YYYY-MM-DDTHH:MM'
        time_of_day = (request.form.get('time_of_day') or '').strip()
        location = (request.form.get('location') or '').strip()
        partners = (request.form.get('partners') or '').strip()
        device = (request.form.get('device') or '').strip()
        vocabulary = (request.form.get('vocabulary') or '').strip()
        trial_window = (request.form.get('trial_window') or '').strip()

        prompting_level = request.form.get('prompting_level', '')
        modeled_words = request.form.get('modeled_words', '')
        used_words = request.form.get('used_words', '')
        observation = request.form.get('observation', '')
        comments = request.form.get('comments', '')

        # Sanitize long text inputs
        for_html = lambda s: bleach.clean(s or "", tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
        observation_clean = for_html(observation)
        comments_clean = for_html(comments)

        # Parse date/time safely
        dt_obj = None
        if dt_str:
            # Accept both "YYYY-MM-DDTHH:MM" and "YYYY-MM-DD"
            try:
                if 'T' in dt_str:
                    dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')
                else:
                    dt_obj = datetime.strptime(dt_str, '%Y-%m-%d')
            except ValueError:
                dt_obj = datetime.utcnow()
        else:
            dt_obj = datetime.utcnow()

        # Store date part in the date column
        date_obj = dt_obj.date()

        # If time_of_day field left blank, use the chosen time (HH:MM) from the datetime field
        if not time_of_day and 'T' in dt_str:
            time_of_day = dt_obj.strftime('%H:%M')

        entry = AACTrialEntry(
            person_name='Tamara Fitzmaurice',
            goal_no=goal_no,
            attainment_level=attainment_level,
            date=date_obj,
            time_of_day=time_of_day,
            location=location,
            partners=partners,
            device=device,
            vocabulary=vocabulary,
            trial_window=trial_window,
            prompting_level=prompting_level,
            modeled_words=modeled_words,
            used_words=used_words,
            observation=observation_clean,
            comments=comments_clean,
            recorded_by=session.get('carer_name')
        )
        db.session.add(entry)
        db.session.commit()
        return jsonify({'success': True, 'id': entry.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/aac/entries', methods=['GET'])
def aac_entries_list():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    # Optional filters
    goal_no = request.args.get('goal_no')
    level = request.args.get('level')
    start = request.args.get('start')  # YYYY-MM-DD
    end   = request.args.get('end')    # YYYY-MM-DD

    q = AACTrialEntry.query.filter_by(person_name='Tamara Fitzmaurice')
    if goal_no:
        q = q.filter(AACTrialEntry.goal_no == int(goal_no))
    if level:
        q = q.filter(AACTrialEntry.attainment_level == int(level))
    if start:
        q = q.filter(AACTrialEntry.date >= datetime.strptime(start, '%Y-%m-%d').date())
    if end:
        q = q.filter(AACTrialEntry.date <= datetime.strptime(end, '%Y-%m-%d').date())

    entries = q.order_by(AACTrialEntry.date.desc(), AACTrialEntry.created_at.desc()).all()
    return render_template('aac_entries.html', entries=entries, PROMPTING_LEVELS=PROMPTING_LEVELS)


@app.route('/aac/export.csv', methods=['GET'])
def aac_export_csv():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    import csv
    from io import StringIO
    from flask import make_response

    # Mirror optional filters (but DO NOT pre-filter by person_name)
    goal_no = request.args.get('goal_no')
    level   = request.args.get('level')
    start   = request.args.get('start')  # YYYY-MM-DD
    end     = request.args.get('end')    # YYYY-MM-DD

    # ✅ You were missing this line:
    q = AACTrialEntry.query

    if goal_no:
        q = q.filter(AACTrialEntry.goal_no == int(goal_no))
    if level:
        q = q.filter(AACTrialEntry.attainment_level == int(level))
    if start:
        q = q.filter(AACTrialEntry.date >= datetime.strptime(start, '%Y-%m-%d').date())
    if end:
        q = q.filter(AACTrialEntry.date <= datetime.strptime(end, '%Y-%m-%d').date())

    rows = q.order_by(AACTrialEntry.date.asc(), AACTrialEntry.created_at.asc()).all()
    print(f"Exporting {len(rows)} AAC rows")  # server log check

    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        'date','time_of_day','location','partners','goal_no','attainment_level',
        'device','vocabulary','trial_window','prompting_level','modeled_words','used_words',
        'observation','comments','recorded_by','created_at'
    ])
    for r in rows:
        writer.writerow([
            r.date.isoformat() if r.date else '',
            r.time_of_day or '',
            r.location or '',
            r.partners or '',
            r.goal_no,
            r.attainment_level,
            r.device or '',
            r.vocabulary or '',
            r.trial_window or '',
            r.prompting_level or '',
            (r.modeled_words or '').replace('\n',' ').strip(),
            (r.used_words or '').replace('\n',' ').strip(),
            (r.observation or '').replace('\n',' ').strip(),
            (r.comments or '').replace('\n',' ').strip(),
            r.recorded_by or '',
            r.created_at.isoformat() if r.created_at else ''
        ])

    csv_data = '\ufeff' + output.getvalue()  # UTF-8 BOM for Excel
    resp = make_response('\ufeff' + csv_data)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=aac_trial_entries.csv"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route('/aac/export_raw.csv', methods=['GET'])
def aac_export_raw_csv():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    import csv
    from io import StringIO
    from flask import make_response

    rows = AACTrialEntry.query.order_by(
        AACTrialEntry.date.asc(), AACTrialEntry.created_at.asc()
    ).all()

    output = StringIO(newline="")
    writer = csv.writer(output)
    # include every column we defined in the model
    writer.writerow([
        'id','person_name','goal_no','attainment_level','date','time_of_day','location',
        'partners','device','vocabulary','trial_window','prompting_level',
        'modeled_words','used_words','observation','comments','recorded_by','created_at'
    ])
    for r in rows:
        writer.writerow([
            r.id,
            r.person_name or '',
            r.goal_no,
            r.attainment_level,
            r.date.isoformat() if r.date else '',
            r.time_of_day or '',
            r.location or '',
            r.partners or '',
            r.device or '',
            r.vocabulary or '',
            r.trial_window or '',
            r.prompting_level or '',
            (r.modeled_words or '').replace('\n',' ').strip(),
            (r.used_words or '').replace('\n',' ').strip(),
            (r.observation or '').replace('\n',' ').strip(),
            (r.comments or '').replace('\n',' ').strip(),
            r.recorded_by or '',
            r.created_at.isoformat() if r.created_at else ''
        ])

    csv_data = '\ufeff' + output.getvalue()
    resp = make_response(csv_data)
    resp.headers["Content-Disposition"] = "attachment; filename=aac_trial_entries_raw.csv"
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


def register_analysis_routes(app: Flask):
    @app.route('/analysis-dashboard', methods=['GET'])
    def analysis_dashboard():
        if 'carer_id' not in session:
            return redirect(url_for('login'))

        activity_counts = (
            db.session.query(
                CareLogEntry.activity_name,
                func.count(CareLogEntry.id).label('entry_count')
            )
            .group_by(CareLogEntry.activity_name)
            .order_by(func.count(CareLogEntry.id).desc())
            .limit(15)
            .all()
        )

        recent_mood = (
            CareLogEntry.query
            .filter(CareLogEntry.activity_name == "Mood")
            .order_by(CareLogEntry.activity_datetime.desc())
            .limit(10)
            .all()
        )

        recent_comm = (
            CareLogEntry.query
            .filter(CareLogEntry.activity_name == "Communication Level")
            .order_by(CareLogEntry.activity_datetime.desc())
            .limit(10)
            .all()
        )

        default_end = date.today()
        default_start = default_end - timedelta(days=29)
        feature_rows = build_daily_features(default_start, default_end)

        return render_template(
            'analysis_dashboard.html',
            activity_counts=activity_counts,
            recent_mood=recent_mood,
            recent_comm=recent_comm,
            feature_rows=feature_rows[-60:]  # Show last 60 days (2 months)
        )

# -------------------------------------------------------------------
# Insights dashboard & feature API
# -------------------------------------------------------------------

def calculate_correlations(start_date=None, end_date=None):
    """Calculate correlations between different metrics."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=89)  # Default to 90 days
    
    # Get daily features
    daily_features = build_daily_features(start_date, end_date)
    
    # Extract metric values aligned by date
    # Store as list of tuples (date, value) to ensure alignment
    metrics_by_date = {}
    for day_data in daily_features:
        date_key = day_data['date']
        metrics_by_date[date_key] = {
            'mood': day_data.get('mood_avg'),
            'sleep': day_data.get('sleep_avg'),
            'communication': day_data.get('communication_avg'),
            'toilet_success_rate': day_data.get('toilet_success_rate'),
            'fluid_intake': day_data.get('fluid_ml'),
            'accidents': day_data.get('accidents'),
            'meltdowns': None,
            'walking': day_data.get('walking_independent_ratio'),
            'motor': day_data.get('motor_avg'),
            'posture': day_data.get('posture_avg')
        }
        # Convert meltdown level to numeric
        if day_data.get('meltdown_level'):
            meltdown_map = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
            metrics_by_date[date_key]['meltdowns'] = meltdown_map.get(day_data['meltdown_level'], 0)
    
    # Extract aligned metric arrays
    metrics = {
        'mood': [],
        'sleep': [],
        'communication': [],
        'toilet_success_rate': [],
        'fluid_intake': [],
        'accidents': [],
        'meltdowns': [],
        'walking': [],
        'motor': [],
        'posture': []
    }
    
    for date_key in sorted(metrics_by_date.keys()):
        day_metrics = metrics_by_date[date_key]
        for metric_name in metrics.keys():
            metrics[metric_name].append(day_metrics[metric_name])
    
    # Calculate Pearson correlation coefficient
    def pearson_correlation(x, y):
        if len(x) != len(y) or len(x) < 2:
            return None
        
        # Filter to days where both metrics have values
        pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
        if len(pairs) < 2:
            return None
        
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        
        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        
        numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
        x_variance = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y_vals[i] - y_mean) ** 2 for i in range(n))
        
        denominator = (x_variance * y_variance) ** 0.5
        if denominator == 0:
            return None
        
        return numerator / denominator
    
    # Calculate all pairwise correlations
    correlation_matrix = {}
    metric_names = list(metrics.keys())
    
    for i, metric1 in enumerate(metric_names):
        correlation_matrix[metric1] = {}
        for metric2 in metric_names:
            if metric1 == metric2:
                correlation_matrix[metric1][metric2] = 1.0
            else:
                corr = pearson_correlation(metrics[metric1], metrics[metric2])
                correlation_matrix[metric1][metric2] = corr
    
    return correlation_matrix, metrics

def analyze_metric_influences(target_metric, threshold='high', start_date=None, end_date=None):
    """Analyze what factors influence a target metric when it's high or low."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=89)
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    
    # Get all log entries in date range
    all_entries = CareLogEntry.query.filter(
        CareLogEntry.activity_datetime >= start_dt,
        CareLogEntry.activity_datetime <= end_dt
    ).order_by(CareLogEntry.activity_datetime.asc()).all()
    
    # Build daily features
    daily_features = build_daily_features(start_date, end_date)
    daily_dict = {f['date']: f for f in daily_features}
    
    # Get target metric values and determine threshold
    target_values = []
    for day_data in daily_features:
        value = None
        if target_metric == 'mood' and day_data.get('mood_avg') is not None:
            value = day_data['mood_avg']
        elif target_metric == 'sleep' and day_data.get('sleep_avg') is not None:
            value = day_data['sleep_avg']
        elif target_metric == 'communication' and day_data.get('communication_avg') is not None:
            value = day_data['communication_avg']
        elif target_metric == 'toilet_success' and day_data.get('toilet_success_rate') is not None:
            value = day_data['toilet_success_rate']
        elif target_metric == 'motor' and day_data.get('motor_avg') is not None:
            value = day_data['motor_avg']
        
        if value is not None:
            target_values.append(value)
    
    if not target_values:
        return {
            'influences': {},
            'carer_frequency': {},
            'food_frequency': {},
            'activity_frequency': {},
            'time_patterns': {},
            'day_of_week_patterns': {}
        }
    
    # Calculate threshold based on 5-way comparison
    sorted_values = sorted(target_values)
    
    # Handle empty case
    if not sorted_values:
        threshold_value = None
        threshold_direction = None
    elif threshold == 'very_high':
        # Top 20% of days (80th percentile)
        idx = min(int(len(sorted_values) * 0.8), len(sorted_values) - 1) if len(sorted_values) > 0 else 0
        threshold_value = sorted_values[idx] if sorted_values else None
        threshold_direction = 'high'
    elif threshold == 'high':
        # Top 40% of days (60th percentile)
        idx = min(int(len(sorted_values) * 0.6), len(sorted_values) - 1) if len(sorted_values) > 0 else 0
        threshold_value = sorted_values[idx] if sorted_values else None
        threshold_direction = 'high'
    elif threshold == 'all':
        # Include all days - no threshold filtering
        threshold_value = None
        threshold_direction = None
    elif threshold == 'low':
        # Bottom 40% of days (40th percentile)
        idx = int(len(sorted_values) * 0.4) if len(sorted_values) > 0 else 0
        threshold_value = sorted_values[idx] if sorted_values else None
        threshold_direction = 'low'
    elif threshold == 'very_low':
        # Bottom 20% of days (20th percentile)
        idx = int(len(sorted_values) * 0.2) if len(sorted_values) > 0 else 0
        threshold_value = sorted_values[idx] if sorted_values else None
        threshold_direction = 'low'
    else:
        # Default to high
        idx = min(int(len(sorted_values) * 0.6), len(sorted_values) - 1) if len(sorted_values) > 0 else 0
        threshold_value = sorted_values[idx] if sorted_values else None
        threshold_direction = 'high'
    
    if threshold_value is None and threshold != 'all':
        threshold_value = 0
    
    # Find days where target metric meets threshold
    target_days = set()
    
    # For communication, also check individual entries if daily average is missing
    if target_metric == 'communication' and threshold == 'all':
        # Include all days that have ANY communication entries
        for entry in all_entries:
            if entry.activity_name == 'Communication Level':
                entry_date = entry.activity_datetime.date().isoformat()
                target_days.add(entry_date)
    
    # Also check daily features (for averages)
    for day_data in daily_features:
        value = None
        if target_metric == 'mood' and day_data.get('mood_avg') is not None:
            value = day_data['mood_avg']
        elif target_metric == 'sleep' and day_data.get('sleep_avg') is not None:
            value = day_data['sleep_avg']
        elif target_metric == 'communication' and day_data.get('communication_avg') is not None:
            value = day_data['communication_avg']
        elif target_metric == 'toilet_success' and day_data.get('toilet_success_rate') is not None:
            value = day_data['toilet_success_rate']
        elif target_metric == 'motor' and day_data.get('motor_avg') is not None:
            value = day_data['motor_avg']
        
        if value is not None:
            if threshold == 'all':
                # Include all days
                target_days.add(day_data['date'])
            elif threshold_direction == 'high' and value >= threshold_value:
                target_days.add(day_data['date'])
            elif threshold_direction == 'low' and value <= threshold_value:
                target_days.add(day_data['date'])
    
    # Analyze factors on target days
    carer_frequency = defaultdict(int)
    food_frequency = defaultdict(int)
    activity_frequency = defaultdict(int)
    time_patterns = defaultdict(int)  # Hour of day
    day_of_week_patterns = defaultdict(int)
    
    # Get entries for target days
    target_entries = []
    for entry in all_entries:
        entry_date = entry.activity_datetime.date().isoformat()
        if entry_date in target_days:
            target_entries.append(entry)
    
    total_target_entries = len(target_entries)
    
    for entry in target_entries:
        # Carer
        if entry.carer_name:
            carer_frequency[entry.carer_name] += 1
        
        # Food
        if entry.activity_name == "Food":
            food_value = entry.value_type or entry.value or ""
            if food_value:
                foods = [f.strip() for f in food_value.split(',')]
                for food in foods:
                    if food:
                        food_frequency[food] += 1
        
        # Activities (excluding the target metric itself)
        if entry.activity_name and entry.activity_name not in ["Mood", "Sleep Quality", "Communication Level", "Toilet Tries", "Fine Motor Skills", "Gross Motor Skills", "Motor Skills"]:
            activity_frequency[entry.activity_name] += 1
        
        # Time patterns
        hour = entry.activity_datetime.hour
        if 6 <= hour < 12:
            time_patterns['Morning (6am-12pm)'] += 1
        elif 12 <= hour < 18:
            time_patterns['Afternoon (12pm-6pm)'] += 1
        elif 18 <= hour < 24:
            time_patterns['Evening (6pm-12am)'] += 1
        else:
            time_patterns['Night (12am-6am)'] += 1
        
        # Day of week
        day_name = entry.activity_datetime.strftime('%A')
        day_of_week_patterns[day_name] += 1
    
    # Calculate percentages
    def calculate_percentages(freq_dict, total):
        if total == 0:
            return {}
        return {k: round((v / total) * 100, 1) for k, v in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)}
    
    # Also analyze other metrics on target days
    influences = {}
    for day_str in target_days:
        day_data = daily_dict.get(day_str)
        if day_data:
            if day_data.get('sleep_avg') is not None:
                influences.setdefault('sleep_avg', []).append(day_data['sleep_avg'])
            if day_data.get('communication_avg') is not None:
                influences.setdefault('communication_avg', []).append(day_data['communication_avg'])
            if day_data.get('toilet_success_rate') is not None:
                influences.setdefault('toilet_success_rate', []).append(day_data['toilet_success_rate'])
            if day_data.get('fluid_ml') is not None:
                influences.setdefault('fluid_ml', []).append(day_data['fluid_ml'])
            if day_data.get('accidents') is not None:
                influences.setdefault('accidents', []).append(day_data['accidents'])
            if day_data.get('motor_avg') is not None:
                influences.setdefault('motor_avg', []).append(day_data['motor_avg'])
            if day_data.get('walking_independent_ratio') is not None:
                influences.setdefault('walking_avg', []).append(day_data['walking_independent_ratio'])
    
    # Calculate averages for other metrics
    metric_averages = {}
    for metric, values in influences.items():
        if values:
            metric_averages[metric] = round(sum(values) / len(values), 2)
    
    # Calculate statistics about the data
    total_entries_count = len(all_entries)
    metric_entries_count = len([e for e in all_entries if 
        (target_metric == 'mood' and e.activity_name == 'Mood') or
        (target_metric == 'sleep' and e.activity_name == 'Sleep Quality') or
        (target_metric == 'communication' and e.activity_name == 'Communication Level') or
        (target_metric == 'toilet_success' and e.activity_name == 'Toilet Tries') or
        (target_metric == 'motor' and e.activity_name in ['Fine Motor Skills', 'Gross Motor Skills', 'Motor Skills'])])
    
    # Calculate min, max, average of target values
    if target_values:
        min_value = min(target_values)
        max_value = max(target_values)
        avg_value = sum(target_values) / len(target_values)
    else:
        min_value = max_value = avg_value = None
    
    # Count total days with data for this metric
    # For communication, count days that have ANY communication entries, not just those with averages
    if target_metric == 'communication':
        # Get unique dates from communication entries
        comm_entry_dates = set()
        for entry in all_entries:
            if entry.activity_name == 'Communication Level':
                comm_entry_dates.add(entry.activity_datetime.date().isoformat())
        total_days_with_data = len(comm_entry_dates)
    else:
        total_days_with_data = len([d for d in daily_features if 
            (target_metric == 'mood' and d.get('mood_avg') is not None) or
            (target_metric == 'sleep' and d.get('sleep_avg') is not None) or
            (target_metric == 'toilet_success' and d.get('toilet_success_rate') is not None) or
            (target_metric == 'motor' and d.get('motor_avg') is not None)])
    
    return {
        'influences': metric_averages,
        'carer_frequency': calculate_percentages(carer_frequency, total_target_entries),
        'food_frequency': calculate_percentages(food_frequency, total_target_entries),
        'activity_frequency': calculate_percentages(activity_frequency, total_target_entries),
        'time_patterns': calculate_percentages(time_patterns, total_target_entries),
        'day_of_week_patterns': calculate_percentages(day_of_week_patterns, total_target_entries),
        'threshold_value': round(threshold_value, 2) if threshold_value is not None else None,
        'threshold_direction': threshold_direction,
        'target_days_count': len(target_days),
        'total_days_with_data': total_days_with_data,
        'total_entries': total_entries_count,
        'metric_entries': metric_entries_count,
        'min_value': round(min_value, 2) if min_value is not None else None,
        'max_value': round(max_value, 2) if max_value is not None else None,
        'avg_value': round(avg_value, 2) if avg_value is not None else None
    }

@app.route('/analysis-dashboard', methods=['GET'])
def analysis_dashboard():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    # Get filter parameters
    target_metric = request.args.get('target_metric', 'mood')
    threshold = request.args.get('threshold', 'all')  # Default to 'all' for broader analysis
    date_range = request.args.get('date_range', '90')
    
    # Check for custom date range
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    if start_date_str and end_date_str:
        # Use custom date range
        try:
            default_start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            default_end = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            # Ensure end date is not in the future
            if default_end > date.today():
                default_end = date.today()
            # Ensure start date is before end date
            if default_start > default_end:
                default_start = default_end - timedelta(days=90)
            date_range = 'custom'
        except (ValueError, TypeError):
            # Fallback to default if parsing fails
            default_end = date.today()
            default_start = default_end - timedelta(days=90)
            date_range = '90'
    else:
        # Use predefined date range
        try:
            date_range_days = int(date_range)
        except (ValueError, TypeError):
            date_range_days = 90
        
        default_end = date.today()
        default_start = default_end - timedelta(days=date_range_days)

    activity_counts = (
        db.session.query(
            CareLogEntry.activity_name,
            func.count(CareLogEntry.id).label('entry_count')
        )
        .group_by(CareLogEntry.activity_name)
        .order_by(func.count(CareLogEntry.id).desc())
        .limit(15)
        .all()
    )

    recent_mood = (
        CareLogEntry.query
        .filter(CareLogEntry.activity_name == "Mood")
        .order_by(CareLogEntry.activity_datetime.desc())
        .limit(10)
        .all()
    )

    recent_comm = (
        CareLogEntry.query
        .filter(CareLogEntry.activity_name == "Communication Level")
        .order_by(CareLogEntry.activity_datetime.desc())
        .limit(10)
        .all()
    )

    # Daily feature snapshot date range (defaults to 60 days, but can be customized)
    snapshot_start_str = request.args.get('snapshot_start_date')
    snapshot_end_str = request.args.get('snapshot_end_date')
    
    if snapshot_start_str and snapshot_end_str:
        try:
            snapshot_start = datetime.strptime(snapshot_start_str, '%Y-%m-%d').date()
            snapshot_end = datetime.strptime(snapshot_end_str, '%Y-%m-%d').date()
            # Ensure end date is not in the future
            if snapshot_end > date.today():
                snapshot_end = date.today()
            # Ensure start date is before end date
            if snapshot_start > snapshot_end:
                snapshot_start = snapshot_end - timedelta(days=59)
        except (ValueError, TypeError):
            # Fallback to default 60 days
            snapshot_end = date.today()
            snapshot_start = snapshot_end - timedelta(days=59)
    else:
        # Default to 60 days
        snapshot_end = date.today()
        snapshot_start = snapshot_end - timedelta(days=59)  # 60 days total
    
    feature_rows = build_daily_features(snapshot_start, snapshot_end)

    # Calculate correlations
    correlation_matrix, metrics_data = calculate_correlations(default_start, default_end)
    
    # Analyze influences for selected metric
    influence_data = analyze_metric_influences(target_metric, threshold, default_start, default_end)

    return render_template(
        'analysis_dashboard.html',
        activity_counts=activity_counts,
        recent_mood=recent_mood,
        recent_comm=recent_comm,
        feature_rows=feature_rows,  # All rows for the selected snapshot date range
        correlation_matrix=correlation_matrix,
        influence_data=influence_data,
        target_metric=target_metric,
        threshold=threshold,
        date_range=date_range,
        start_date=default_start.isoformat() if 'default_start' in locals() else None,
        end_date=default_end.isoformat() if 'default_end' in locals() else None,
        snapshot_start_date=snapshot_start.isoformat(),
        snapshot_end_date=snapshot_end.isoformat(),
        metrics_data=metrics_data
    )


@app.route('/analysis/data', methods=['GET'])
def analysis_data():
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # Check for snapshot-specific dates first (for the daily feature snapshot)
    snapshot_start_param = request.args.get('snapshot_start')
    snapshot_end_param = request.args.get('snapshot_end')
    
    if snapshot_start_param and snapshot_end_param:
        # Use snapshot dates
        try:
            start_date = datetime.strptime(snapshot_start_param, '%Y-%m-%d').date()
            end_date = datetime.strptime(snapshot_end_param, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid snapshot date format. Use YYYY-MM-DD.'}), 400
    else:
        # Fall back to regular start/end params
        start_param = request.args.get('start')
        end_param = request.args.get('end')

        try:
            end_date = datetime.strptime(end_param, '%Y-%m-%d').date() if end_param else date.today()
        except ValueError:
            return jsonify({'error': 'Invalid end date format. Use YYYY-MM-DD.'}), 400

        try:
            start_date = datetime.strptime(start_param, '%Y-%m-%d').date() if start_param else end_date - timedelta(days=59)  # 60 days (2 months)
        except ValueError:
            return jsonify({'error': 'Invalid start date format. Use YYYY-MM-DD.'}), 400

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    features = build_daily_features(start_date, end_date)
    return jsonify({'features': features})


@app.route('/analysis', methods=['GET', 'POST'])
def add_analysis():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Read raw HTML from TinyMCE and sanitize it
            raw_html = request.form['analysis_text']
            if not raw_html.strip():
                flash('Analysis text cannot be empty!', 'error')
                return redirect(url_for('add_analysis'))

            clean_html = bleach.clean(
                raw_html,
                tags=ALLOWED_TAGS,
                attributes=ALLOWED_ATTRS,
                strip=True
            )

            entry = AnalysisEntry(
                carer_id=session['carer_id'],
                carer_name=session['carer_name'],
                analysis_text=clean_html   # <- store sanitized HTML
            )
            db.session.add(entry)
            db.session.commit()
            flash('Analysis added successfully!', 'success')
            return redirect(url_for('view_analyses'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding analysis: {str(e)}', 'error')
            return redirect(url_for('add_analysis'))

    return render_template('analysis_entry.html')

@app.route('/view-analyses', methods=['GET'])
def view_analyses():
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    # Get filter parameters
    carer_name = request.args.get('carer_name')

    # Build query
    query = AnalysisEntry.query
    if carer_name:
        query = query.filter(AnalysisEntry.carer_name == carer_name)

    analyses = query.order_by(AnalysisEntry.created_at.desc()).all()

    # Get unique carer names for filter dropdown
    carer_names = db.session.query(AnalysisEntry.carer_name).distinct().all()
    carer_names = [name[0] for name in carer_names]

    return render_template(
        'view_analyses.html',
        analyses=analyses,
        carer_names=carer_names,
        selected_carer=carer_name
    )


# Helper function to get unread notice count for current user
def get_unread_notice_count():
    if 'carer_id' not in session:
        return 0
    user_id = session['carer_id']
    # Get all notices
    all_notices = Notice.query.all()
    # Get notices this user has read
    read_notices = NoticeRead.query.filter_by(user_id=user_id).all()
    read_notice_ids = {nr.notice_id for nr in read_notices}
    # Count unread notices
    unread_count = sum(1 for notice in all_notices if notice.id not in read_notice_ids)
    return unread_count

# Make function available to all templates
@app.context_processor
def inject_notice_count():
    return dict(get_unread_notice_count=get_unread_notice_count)

@app.route('/notice-board', methods=['GET'])
def notice_board():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    is_admin = user.is_admin if user else False
    
    # Get all notices, newest first
    notices = Notice.query.order_by(Notice.created_at.desc()).all()
    
    # Get which notices this user has read
    read_notices = NoticeRead.query.filter_by(user_id=user_id).all()
    read_notice_ids = {nr.notice_id for nr in read_notices}
    
    # Separate notices into unread and read lists
    unread_notices = [n for n in notices if n.id not in read_notice_ids]
    read_notices_list = [n for n in notices if n.id in read_notice_ids]
    
    # Don't auto-mark as read - let user see them first
    # Notices will be marked as read via JavaScript when scrolled into view
    
    return render_template('notice_board.html', 
                         notices=notices, 
                         unread_notices=unread_notices,
                         read_notices=read_notices_list,
                         read_notice_ids=read_notice_ids, 
                         user_id=user_id, 
                         is_admin=is_admin)

@app.route('/notice-board/add', methods=['GET', 'POST'])
def add_notice():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        is_important = request.form.get('is_important') == 'on'
        
        if title and content:
            notice = Notice(
                title=title,
                content=content,
                created_by=session['carer_id'],
                is_important=is_important
            )
            db.session.add(notice)
            db.session.flush()  # Get the notice ID
            
            # Automatically mark notice as read for the creator (they've already seen it)
            notice_read = NoticeRead(notice_id=notice.id, user_id=session['carer_id'])
            db.session.add(notice_read)
            db.session.commit()
            flash('Notice added successfully!', 'success')
            return redirect(url_for('notice_board'))
        else:
            flash('Title and content are required!', 'error')
    
    return render_template('add_notice.html')

@app.route('/notice-board/<int:notice_id>/reply', methods=['POST'])
def reply_to_notice(notice_id):
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    notice = Notice.query.get_or_404(notice_id)
    content = request.form.get('reply_content', '').strip()
    
    if content:
        reply = NoticeReply(
            notice_id=notice_id,
            content=content,
            created_by=session['carer_id']
        )
        db.session.add(reply)
        db.session.commit()
        flash('Reply added successfully!', 'success')
    else:
        flash('Reply content cannot be empty!', 'error')
    
    return redirect(url_for('notice_board'))

@app.route('/notice-board/<int:notice_id>/mark-read', methods=['POST'])
def mark_notice_read(notice_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['carer_id']
    
    # Check if already read
    existing_read = NoticeRead.query.filter_by(notice_id=notice_id, user_id=user_id).first()
    if not existing_read:
        notice_read = NoticeRead(notice_id=notice_id, user_id=user_id)
        db.session.add(notice_read)
        db.session.commit()
    
    return jsonify({'success': True})

@app.route('/notice-board/<int:notice_id>/mark-unread', methods=['POST'])
def mark_notice_unread(notice_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_id = session['carer_id']
    
    # Remove the read record
    existing_read = NoticeRead.query.filter_by(notice_id=notice_id, user_id=user_id).first()
    if existing_read:
        db.session.delete(existing_read)
        db.session.commit()
    
    return jsonify({'success': True})

@app.route('/notice-board/<int:notice_id>/delete', methods=['POST'])
def delete_notice(notice_id):
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Check if user is admin
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    notice = Notice.query.get_or_404(notice_id)
    
    # Delete the notice (cascade will handle replies and reads)
    db.session.delete(notice)
    db.session.commit()
    
    flash('Notice deleted successfully!', 'success')
    return jsonify({'success': True})

if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'static/uploads_test'), exist_ok=True)
    with app.app_context():
        db.create_all()
        add_missing_columns()
        add_user_profile_columns()
        seed_gas_goals()  # NEW: seed Tamara's GAS goals once

    app.run(host='0.0.0.0', port=5001, debug=True)