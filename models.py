from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class CareLogEntry(db.Model):
    __tablename__ = 'care_log_entries'

    id = db.Column(db.Integer, primary_key=True)
    carer_id = db.Column(db.String(100), nullable=False)
    carer_name = db.Column(db.String(100), nullable=False)
    activity_name = db.Column(db.String(100), nullable=False)
    activity_type = db.Column(db.String(50), nullable=True)
    value = db.Column(db.String(100), nullable=False)
    value_type = db.Column(db.String(100), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    activity_datetime = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    media = db.relationship('Media', backref='entry', lazy=True, cascade="all, delete-orphan")

class Media(db.Model):
    __tablename__ = 'media'
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(255), nullable=False)
    entry_id = db.Column(db.Integer, db.ForeignKey('care_log_entries.id'), nullable=False)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=False)

class AnalysisEntry(db.Model):
    __tablename__ = 'analysis_entries'
    id = db.Column(db.Integer, primary_key=True)
    carer_id = db.Column(db.String(100), nullable=False)
    carer_name = db.Column(db.String(100), nullable=False)
    analysis_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- NEW: GAS goals + AAC trial entries ---

class GASGoal(db.Model):
    __tablename__ = 'gas_goals'

    id = db.Column(db.Integer, primary_key=True)
    person_name = db.Column(db.String(120), nullable=False, default='Tamara Fitzmaurice')
    goal_no = db.Column(db.Integer, nullable=False)  # 1,2,3
    competency = db.Column(db.String(120), nullable=False)  # e.g., Linguistic/Social, Operational, Strategic/Social

    baseline_text = db.Column(db.Text, nullable=False)  # -2
    minus1_text = db.Column(db.Text, nullable=False)    # -1
    zero_text = db.Column(db.Text, nullable=False)      # 0
    plus1_text = db.Column(db.Text, nullable=False)     # +1
    plus2_text = db.Column(db.Text, nullable=False)     # +2

    # optional short name/label for display
    short_label = db.Column(db.String(200), nullable=True)


class AACTrialEntry(db.Model):
    __tablename__ = 'aac_trial_entries'

    id = db.Column(db.Integer, primary_key=True)

    # Link to a GASGoal by (goal_no + person_name) rather than FK to keep this simple/portable
    person_name = db.Column(db.String(120), nullable=False, default='Tamara Fitzmaurice')
    goal_no = db.Column(db.Integer, nullable=False)  # 1,2,3

    # GAS rating -2 .. +2
    attainment_level = db.Column(db.Integer, nullable=False)  # -2..+2

    # Diary / context
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    time_of_day = db.Column(db.String(50), nullable=True)      # Morning / Afternoon / Evening (free text)
    location = db.Column(db.String(120), nullable=True)        # Home / Therapy / School / etc.
    partners = db.Column(db.String(255), nullable=True)        # Support people

    device = db.Column(db.String(120), nullable=True)          # Device name
    vocabulary = db.Column(db.String(120), nullable=True)      # Vocab set/page
    trial_window = db.Column(db.String(120), nullable=True)    # e.g., "Week 2", "2025-10-06"

    # Prompting & observations
    prompting_level = db.Column(db.String(120), nullable=True) # Independent / Pause / Facial / Body language / Comment / Visual
    modeled_words = db.Column(db.Text, nullable=True)
    used_words = db.Column(db.Text, nullable=True)
    observation = db.Column(db.Text, nullable=True)
    comments = db.Column(db.Text, nullable=True)

    # Audit
    recorded_by = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
