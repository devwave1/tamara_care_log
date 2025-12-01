from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, date, timedelta
from models import db, CareLogEntry, User, Media, AnalysisEntry
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
    if not value:
        return 0.0
    digits = ''.join(ch for ch in str(value) if ch.isdigit() or ch == '.')
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


def _posture_score(*texts):
    for text in texts:
        if not text:
            continue
        text = str(text).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            lowered = text.lower()
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
        "health_events": 0,
        "walking_scores": [],
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
            if value.lower().startswith("yes"):
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
            score = COMM_MAP.get(value) or COMM_MAP.get(value_type)
            if score is not None:
                bucket["communication_scores"].append(score)

        elif name == "Fluid Intake":
            ml = _extract_ml(value) or _extract_ml(value_type)
            bucket["fluid_ml"] += ml

        elif name == "Food":
            sources = []
            if value:
                sources.append(value)
            if value_type:
                sources.extend(part.strip() for part in value_type.split(',') if part.strip())
            for source in sources:
                bucket["food_types"].add(source)

        elif name == "Meltdowns":
            score = MELTDOWN_MAP.get(value) or MELTDOWN_MAP.get(value_type)
            if score is not None:
                bucket["meltdown_scores"].append(score)

        elif name == "Menstrual Cycle":
            score = MENSTRUAL_MAP.get(value) or MENSTRUAL_MAP.get(value_type)
            if score:
                bucket["menstrual_levels"].append(score)

        elif name == "Health / Medical":
            bucket["health_events"] += 1

        elif name == "Walking Ability":
            score = WALKING_MAP.get(value) or WALKING_MAP.get(value_type)
            if score is not None:
                bucket["walking_scores"].append(score)

        elif "posture" in name.lower():
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
            "health_events": bucket["health_events"],
            "walking_independent_ratio": walking_ratio,
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
                    if also_prompt_log and (aac_prompt or aac_level is not None or aac_words):
                        prompt_notes_bits = []
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
            flash('Login successful!', 'success')
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
        activity_data[l.activity_name][label] = float(l.value)
        tooltip_data[l.activity_name][label] = {
            'value_type': l.value_type or '',
            'notes': l.notes or '',
            'duration': l.duration or ''
        }

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'teal', 'brown', 'gray']

    event_based = ["Toilet Tries", "Accidents", "Meltdowns", "Teeth Brushed", "Bath/Shower Completed", "Nap Taken", "Food", "Sensory Sensitivities", "Message for Careers", "Menstrual Cycle", "Health / Medical"]
    line_based = ["Mood", "Sleep Quality", "Communication Level", "Walking Ability", "Fine Motor Skills", "Gross Motor Skills", "Fluid Intake", "Exercise / Physical Activity"]

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
    return render_template('admin.html', users=users)

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
        "Fine Motor Skills": ["Poor", "Fair", "Good"],
        "Gross Motor Skills": ["Poor", "Fair", "Good"],
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
            feature_rows=feature_rows[-14:]
        )

# -------------------------------------------------------------------
# Insights dashboard & feature API
# -------------------------------------------------------------------
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
        feature_rows=feature_rows[-14:]
    )


@app.route('/analysis/data', methods=['GET'])
def analysis_data():
    if 'carer_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    start_param = request.args.get('start')
    end_param = request.args.get('end')

    try:
        end_date = datetime.strptime(end_param, '%Y-%m-%d').date() if end_param else date.today()
    except ValueError:
        return jsonify({'error': 'Invalid end date format. Use YYYY-MM-DD.'}), 400

    try:
        start_date = datetime.strptime(start_param, '%Y-%m-%d').date() if start_param else end_date - timedelta(days=29)
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


if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'static/uploads_test'), exist_ok=True)
    with app.app_context():
        db.create_all()
        add_missing_columns()
        seed_gas_goals()  # NEW: seed Tamara’s GAS goals once

    app.run(host='0.0.0.0', port=5001, debug=True)