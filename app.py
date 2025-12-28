from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, date, timedelta
from models import db, CareLogEntry, User, Media, AnalysisEntry, AACTrialEntry, Notice, NoticeReply, NoticeRead, CalendarEvent, InsuranceRecord, FocusTask, FocusStage, FocusEntry, Plan, Tenant, PlanFeature, DirectMessage, DirectMessageRecipient, DirectMessageReply, DirectMessageRead, DirectMessageReplyRead, Activity, ActivityValue, QuickPickButton, TenantAccess, TenantInvitation, TenantAccessRequest, Service, Schedule, Shift, ShiftGuest, ShiftSwapRequest, ShiftReminder, ShiftHistory, ShiftPayment
from flask_bcrypt import Bcrypt
from utils import sync_to_google_sheets
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from sqlalchemy.sql import text
from sqlalchemy import func, or_
from collections import defaultdict
import bleach
from flask import abort
from flask import make_response
import json
import time
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


app = Flask(__name__)

# Timezone conversion helper
def utc_to_melbourne(utc_dt):
    """Convert UTC datetime to Australia/Melbourne timezone"""
    if utc_dt is None:
        return None
    try:
        from datetime import timedelta
        # Melbourne is UTC+10 (AEST) or UTC+11 (AEDT during daylight saving)
        # December is summer in Australia, so use UTC+11 (AEDT)
        # If the datetime is naive, assume it's stored in UTC and add 11 hours
        if isinstance(utc_dt, datetime):
            melbourne_offset = timedelta(hours=11)  # AEDT for summer months
            return utc_dt + melbourne_offset
        return utc_dt
    except Exception as e:
        print(f"Error converting timezone: {e}")
        return utc_dt

@app.template_filter('to_melbourne')
def to_melbourne_filter(dt):
    """Jinja2 filter to convert UTC datetime to Melbourne time"""
    return utc_to_melbourne(dt)

@app.template_filter('melbourne_time')
def melbourne_time_filter(dt):
    """Jinja2 filter to convert UTC datetime to Melbourne time and format"""
    melbourne_dt = utc_to_melbourne(dt)
    if melbourne_dt:
        return melbourne_dt.strftime('%d %b %Y at %I:%M %p')
    return dt.strftime('%d %b %Y at %I:%M %p') if dt else ''

@app.template_filter('melbourne_time_short')
def melbourne_time_short_filter(dt):
    """Jinja2 filter to convert UTC datetime to Melbourne time and format (short, no 'at')"""
    melbourne_dt = utc_to_melbourne(dt)
    if melbourne_dt:
        return melbourne_dt.strftime('%d %b %Y %I:%M %p')
    return dt.strftime('%d %b %Y %I:%M %p') if dt else ''

@app.template_filter('nl2br')
def nl2br_filter(text):
    """Convert newlines to HTML line breaks"""
    if not text:
        return ''
    from markupsafe import Markup, escape
    # Escape HTML first, then convert newlines to <br>
    escaped = escape(str(text))
    return Markup(escaped.replace('\n', '<br>\n'))

# Email utility function
def send_email(subject, body, to_email=None, html_body=None):
    """
    Send an email using SMTP configuration from environment variables.
    Returns True if sent successfully, False otherwise.
    """
    # Check if email is enabled (default to True for convenience, can be disabled via env var)
    mail_enabled = os.environ.get('MAIL_ENABLED', 'True').lower() == 'true'
    if not mail_enabled:
        print(f"[Email disabled] Would send: {subject} to {to_email}")
        return False
    
    # Get email configuration
    # SECURITY NOTE: Defaults are for convenience. For production, use environment variables.
    mail_server = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    mail_port = int(os.environ.get('MAIL_PORT', 587))
    mail_use_tls = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    mail_use_ssl = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'
    mail_username = os.environ.get('MAIL_USERNAME', 'fitzmauricetamara@gmail.com')
    mail_password = os.environ.get('MAIL_PASSWORD', 'rbbdlzwquylaxpno')  # Gmail App Password
    mail_from = os.environ.get('MAIL_FROM', 'fitzmauricetamara@gmail.com')
    mail_to = to_email or os.environ.get('MAIL_TO', 'tomf@wwave.com.au')
    
    # Validate configuration
    if not mail_server or not mail_username or not mail_password:
        print(f"[Email error] Missing SMTP configuration (server, username, or password)")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = mail_from
        msg['To'] = mail_to
        msg['Subject'] = subject
        
        # Add text and HTML parts
        if html_body:
            part1 = MIMEText(body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server and send
        if mail_use_ssl:
            server = smtplib.SMTP_SSL(mail_server, mail_port)
        else:
            server = smtplib.SMTP(mail_server, mail_port)
            if mail_use_tls:
                server.starttls()
        
        server.login(mail_username, mail_password)
        server.send_message(msg)
        server.quit()
        
        print(f"[Email sent] {subject} to {mail_to}")
        return True
        
    except Exception as e:
        print(f"[Email error] Failed to send email: {str(e)}")
        return False

# ============================================================================
# Feature Flag Helper Functions
# ============================================================================

def get_tenant_id_for_user(user_id):
    """
    Get tenant_id for a user.
    For single-tenant: always returns 1
    For multi-tenant: looks up user's tenant_id from users table
    """
    # Single-tenant implementation: always return 1
    # When enabling multi-tenant, update users table to include tenant_id
    # and query: User.query.get(user_id).tenant_id
    return 1


def check_feature_enabled(tenant_id, feature_key):
    """
    Check if a feature is enabled for a tenant's plan.
    
    Args:
        tenant_id: Tenant ID (use get_tenant_id_for_user() to get from user)
        feature_key: Feature flag key (e.g., 'messaging.family', 'booking.manage')
    
    Returns:
        bool: True if feature is enabled, False otherwise
    """
    tenant = Tenant.query.get(tenant_id)
    if not tenant or not tenant.plan:
        return False
    
    feature = PlanFeature.query.filter_by(
        plan_id=tenant.plan_id,
        feature_key=feature_key
    ).first()
    
    if not feature:
        return False
    
    return feature.feature_value.lower() == 'true'


def get_feature_limit(tenant_id, feature_key):
    """
    Get numeric limit for a feature (e.g., max custom activities).
    
    Args:
        tenant_id: Tenant ID
        feature_key: Feature flag key (e.g., 'activities.custom_limit')
    
    Returns:
        int: Limit value, or -1 for unlimited, or 0 if not found/disabled
    """
    tenant = Tenant.query.get(tenant_id)
    if not tenant or not tenant.plan:
        return 0
    
    feature = PlanFeature.query.filter_by(
        plan_id=tenant.plan_id,
        feature_key=feature_key
    ).first()
    
    if not feature:
        return 0
    
    # Handle "unlimited" convention: store -1 in database for unlimited
    if feature.feature_value.lower() in ('unlimited', '-1'):
        return -1
    
    try:
        limit = int(feature.feature_value)
        return limit if limit >= 0 else -1  # Treat negative as unlimited
    except (ValueError, TypeError):
        return 0


def is_unlimited(limit_value):
    """
    Helper to check if a limit value means unlimited.
    
    Args:
        limit_value: Limit value from get_feature_limit()
    
    Returns:
        bool: True if unlimited, False otherwise
    """
    return limit_value == -1 or limit_value is None


def check_limit_not_exceeded(tenant_id, feature_key, current_count):
    """
    Check if current count is within feature limit.
    
    Args:
        tenant_id: Tenant ID
        feature_key: Feature flag key (e.g., 'activities.custom_limit')
        current_count: Current count of items
    
    Returns:
        tuple: (is_allowed: bool, limit: int, remaining: int)
    """
    limit = get_feature_limit(tenant_id, feature_key)
    
    if is_unlimited(limit):
        return (True, -1, -1)  # Unlimited: always allowed
    
    if limit == 0:
        return (False, 0, 0)  # Feature disabled
    
    remaining = limit - current_count
    is_allowed = remaining > 0
    
    return (is_allowed, limit, remaining)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.abspath('instance/care_log_v1.sqlite')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads_test')
app.config['TESTING'] = True
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_super_secret_key'

# Google Calendar for Tamara's schedule
# Get embed URL from: Google Calendar → Settings → Integrate calendar → Copy "Public URL to embed"
app.config['TAMARA_CALENDAR_URL'] = os.environ.get('TAMARA_CALENDAR_URL', 'https://calendar.google.com/calendar/embed?src=fitzmauricetamara%40gmail.com&ctz=Australia%2FMelbourne')

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

def parse_date_field(value):
    """Parse YYYY-MM-DD date strings to date objects; return None on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (ValueError, TypeError):
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
    """Check if current user is admin (super_user or admin role, or legacy is_admin flag)"""
    u = User.query.filter_by(username=session.get('carer_id')).first()
    if not u:
        return False
    # Super users and admins both have admin access
    user_role = (u.role or '').lower()
    return user_role in ('super_user', 'admin') or u.is_admin

def current_user_is_super_user():
    """Check if current user is super user (dev team / highest level)"""
    u = User.query.filter_by(username=session.get('carer_id')).first()
    return u and (u.role or '').lower() == 'super_user'

def current_user_role():
    """Get current user's role"""
    u = User.query.filter_by(username=session.get('carer_id')).first()
    if not u:
        return 'guest'
    # Legacy: if is_admin flag is set but role is not, return 'admin'
    if u.is_admin and not u.role:
        return 'admin'
    return (u.role or 'carer').lower()

def current_user_is_clinician():
    """Check if current user is clinician/professional (backward compatible)"""
    role = current_user_role().lower()
    return role in ('clinician', 'professional', 'admin', 'super_user')

def current_user_is_professional():
    """Check if current user is a professional (OT, Speech, Behavioral, etc.)"""
    u = User.query.filter_by(username=session.get('carer_id')).first()
    return u and (u.role or '').lower() == 'professional'

def current_user_is_readonly():
    """Check if current user has read-only access"""
    u = User.query.filter_by(username=session.get('carer_id')).first()
    if not u:
        return False
    return (u.role or '').lower() == 'readonly' or u.is_readonly

def current_user_can_edit():
    """Check if current user can edit (not read-only)"""
    return not current_user_is_readonly()

def can_edit_analysis(entry):
    return entry.carer_id == session.get('carer_id') or current_user_is_admin()

def get_current_user():
    """Get the current logged-in user object"""
    if 'carer_id' not in session:
        return None
    return User.query.filter_by(username=session['carer_id']).first()

def get_current_tenant_id():
    """Get the current tenant ID for the logged-in user"""
    user = get_current_user()
    if not user:
        return None
    # For now, single-tenant: always return 1
    # In future, could be: return user.tenant_id or get_tenant_id_for_user(user.id)
    return 1

def get_accessible_tenants_for_org_user(user_id):
    """Get all tenants that an organization user has access to"""
    from models import TenantAccess
    accesses = TenantAccess.query.filter_by(
        organization_user_id=user_id,
        is_active=True
    ).all()
    return [access.tenant for access in accesses]


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
        care_columns = [col['name'] for col in inspector.get_columns('care_log_entries')]
        if 'value_type' not in care_columns:
            db.session.execute(text('ALTER TABLE care_log_entries ADD COLUMN value_type VARCHAR(100)'))
            db.session.commit()
            print("Added 'value_type' column to care_log_entries table.")
        if 'focus_task_id' not in care_columns:
            db.session.execute(text('ALTER TABLE care_log_entries ADD COLUMN focus_task_id INTEGER'))
            db.session.commit()
            print("Added 'focus_task_id' column to care_log_entries table.")

        user_columns = [col['name'] for col in inspector.get_columns('users')]
        if 'role' not in user_columns:
            db.session.execute(text('ALTER TABLE users ADD COLUMN role VARCHAR(50) DEFAULT "carer"'))
            db.session.commit()
            print("Added 'role' column to users table.")
        
        # Check focus_tasks table for auto_capture_activities column
        try:
            focus_task_columns = [col['name'] for col in inspector.get_columns('focus_tasks')]
            if 'auto_capture_activities' not in focus_task_columns:
                db.session.execute(text('ALTER TABLE focus_tasks ADD COLUMN auto_capture_activities TEXT'))
                db.session.commit()
                print("Added 'auto_capture_activities' column to focus_tasks table.")
        except Exception as e:
            print(f"Note: Could not check focus_tasks table: {e}")
        if 'reset_token' not in user_columns:
            db.session.execute(text('ALTER TABLE users ADD COLUMN reset_token VARCHAR(100)'))
            db.session.commit()
            print("Added 'reset_token' column to users table.")
        if 'reset_token_expiry' not in user_columns:
            db.session.execute(text('ALTER TABLE users ADD COLUMN reset_token_expiry DATETIME'))
            db.session.commit()
            print("Added 'reset_token_expiry' column to users table.")

def seed_focus_stages():
    """Ensure the five fixed focus stages exist."""
    from models import FocusStage
    stages = [
        ("Q", "WHY", "Clinical Rationale", 1),
        ("W", "WHAT", "Defined Focus Task", 2),
        ("H", "HOW", "Implementation Method", 3),
        ("C", "CHECK", "Evidence & Feedback", 4),
        ("Z", "AUTO", "Automate (Independence)", 5),
    ]
    for code, short, name, order in stages:
        if not FocusStage.query.get(code):
            db.session.add(FocusStage(code=code, short_name=short, name=name, order=order))
    db.session.commit()

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
    seed_focus_stages()

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
            focus_task_id_raw = request.form.get('focus_task_id')
            focus_task_id = None
            try:
                focus_task_id = int(focus_task_id_raw) if focus_task_id_raw else None
            except (TypeError, ValueError):
                focus_task_id = None
            focus_as_check = request.form.get('focus_as_check') == 'on'

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
            activity_date = activity_dt.date()

            # Auto-detect matching focused tasks if not manually selected
            if not focus_task_id:
                import json
                # Find active focused tasks that:
                # 1. Are active
                # 2. Have the current activity in their auto_capture_activities
                # 3. Current date is within focus_start and focus_end window
                active_tasks = FocusTask.query.filter_by(status='active').all()
                for task in active_tasks:
                    # Check date range
                    if task.focus_start and task.focus_end:
                        if not (task.focus_start <= activity_date <= task.focus_end):
                            continue
                    elif task.focus_start:
                        if activity_date < task.focus_start:
                            continue
                    elif task.focus_end:
                        if activity_date > task.focus_end:
                            continue
                    
                    # Check if activity matches auto-capture list
                    if task.auto_capture_activities:
                        try:
                            auto_activities = json.loads(task.auto_capture_activities)
                            if activity_name in auto_activities:
                                focus_task_id = task.id
                                focus_as_check = True  # Auto-capture always creates CHECK entry
                                break  # Use first matching task
                        except (json.JSONDecodeError, TypeError):
                            pass

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
                activity_type='manual',
                focus_task_id=focus_task_id
            )
            db.session.add(log)
            db.session.commit()

            # If linked to a focus and user opted to record as CHECK, add a FocusEntry (stage C)
            if focus_task_id and focus_as_check:
                task = FocusTask.query.get(focus_task_id)
                if task:
                    entry_title = f"Log: {activity_name}"
                    detail_parts = []
                    if value:
                        detail_parts.append(f"Value: {value}")
                    if value_type:
                        detail_parts.append(f"Value type: {value_type}")
                    if request.form.get('notes'):
                        detail_parts.append(f"Notes: {request.form.get('notes')}")
                    detail_text = " | ".join(detail_parts) if detail_parts else "Logged activity."
                    db.session.add(FocusEntry(
                        task_id=focus_task_id,
                        stage_code='C',
                        title=entry_title,
                        detail=detail_text,
                        added_by=session.get('carer_id', 'unknown')
                    ))
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
    active_focuses = FocusTask.query.filter_by(status='active').order_by(FocusTask.title.asc()).all()
    
    # Load activities from database (fallback to empty list if table doesn't exist yet)
    try:
        activities = Activity.query.filter_by(is_active=True).order_by(Activity.display_order, Activity.name).all()
        quick_picks = QuickPickButton.query.filter_by(is_active=True).order_by(QuickPickButton.display_order).all()
    except Exception as e:
        # If tables don't exist yet, use empty lists (migration hasn't been run)
        print(f"Warning: Could not load activities from database: {e}")
        activities = []
        quick_picks = []
    
    # Prepare focus tasks data with auto-capture activities for JavaScript
    import json
    focus_tasks_data = []
    for ft in active_focuses:
        auto_activities = []
        if ft.auto_capture_activities:
            try:
                auto_activities = json.loads(ft.auto_capture_activities)
            except (json.JSONDecodeError, TypeError):
                pass
        focus_tasks_data.append({
            'id': ft.id,
            'title': ft.title,
            'short_code': ft.short_code,
            'focus_start': ft.focus_start.strftime('%Y-%m-%d') if ft.focus_start else None,
            'focus_end': ft.focus_end.strftime('%Y-%m-%d') if ft.focus_end else None,
            'auto_capture_activities': auto_activities
        })
    
    # Ensure we pass an empty array if no focus tasks to avoid JSON errors
    focus_tasks_data_json = json.dumps(focus_tasks_data) if focus_tasks_data else '[]'
    
    return render_template('log_entry.html', 
                         current_datetime=now, 
                         focus_tasks=active_focuses,
                         focus_tasks_data=focus_tasks_data_json,
                         activities=activities,
                         quick_picks=quick_picks)


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
            
            # Check if there's a stored invitation code to process
            if 'invitation_code' in session:
                invitation_code = session.pop('invitation_code')
                return redirect(url_for('join_tenant_via_invitation', invitation_code=invitation_code))
            
            # Check for unread notices
            unread_count = get_unread_notice_count()
            if unread_count > 0:
                flash(f'You have {unread_count} unread notice{"s" if unread_count > 1 else ""}! Check the Notice Board.', 'info')
            return redirect(url_for('log_entry'))
        flash('Invalid credentials or account not approved!', 'error')
    return render_template('login.html')

def validate_username(username):
    """Validate username according to rules"""
    if not username:
        return False, "Username is required"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 50:
        return False, "Username must be 50 characters or less"
    # Allow letters, numbers, underscores, hyphens, and dots
    if not username.replace('_', '').replace('-', '').replace('.', '').isalnum():
        return False, "Username can only contain letters, numbers, underscores, hyphens, and dots"
    # Must start with a letter or number
    if not username[0].isalnum():
        return False, "Username must start with a letter or number"
    return True, ""

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        
        # Validate username
        is_valid, error_msg = validate_username(username)
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists!', 'error')
            return redirect(url_for('register'))
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=password_hash, is_approved=False)
        db.session.add(new_user)
        db.session.commit()
        
        # Get all admin users for message and email notifications
        admin_users = User.query.filter(
            or_(
                User.is_admin == True,
                User.role == 'super_user',
                User.role == 'admin'
            )
        ).all()
        
        # Create a message on the messages board about the new registration (sent to all admins)
        if admin_users:
            # Use the first admin as the sender (system message)
            sender_admin = admin_users[0]
            tenant_id = get_tenant_id_for_user(sender_admin.id)
            
            message_subject = f"New User Registration: {username}"
            message_content = f"""A new user has registered and is awaiting approval:

**Username:** {username}
**Email:** {email}
**Registration Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Please review and approve this user in the Admin Panel."""
            
            # Create the message
            new_message = DirectMessage(
                subject=message_subject,
                content=message_content,
                sender_id=sender_admin.id,
                recipient_id=admin_users[0].id,  # First recipient for backward compatibility
                tenant_id=tenant_id,
                message_type='general',
                is_urgent=True  # Mark as urgent so admins see it
            )
            db.session.add(new_message)
            db.session.flush()  # Get message ID
            
            # Add all admins as recipients
            for admin in admin_users:
                msg_recipient = DirectMessageRecipient(
                    message_id=new_message.id,
                    recipient_id=admin.id
                )
                db.session.add(msg_recipient)
            
            db.session.commit()
        
        # Send email to the new user
        notice_board_url = url_for('notice_board', _external=True)
        user_subject = "Welcome to Tamara Care Log - Registration Successful"
        user_body = f"""
Hello {username},

Thank you for registering with Tamara Care Log!

Your account has been created and is currently pending admin approval. Once an administrator approves your account, you will be able to log in and start using the system.

**Important:** Please check the Notice Board regularly for updates and important announcements:
{notice_board_url}

You will receive an email notification once your account has been approved.

If you have any questions, please contact your administrator.

---
Tamara Care Log
"""
        user_html_body = f"""
<html>
<body>
<h2>Welcome to Tamara Care Log!</h2>
<p>Hello <strong>{username}</strong>,</p>
<p>Thank you for registering with Tamara Care Log!</p>
<p>Your account has been created and is currently <strong>pending admin approval</strong>. Once an administrator approves your account, you will be able to log in and start using the system.</p>
<p><strong>📋 Important:</strong> Please check the <a href="{notice_board_url}" style="color: #0d6efd;">Notice Board</a> regularly for updates and important announcements.</p>
<p>You will receive an email notification once your account has been approved.</p>
<p>If you have any questions, please contact your administrator.</p>
<hr>
<p><small>Tamara Care Log</small></p>
</body>
</html>
"""
        try:
            send_email(user_subject, user_body, to_email=email, html_body=user_html_body)
        except Exception as e:
            print(f"Error sending registration email to user: {e}")
        
        # Send emails to all admins
        messages_url = url_for('messages', _external=True)
        admin_panel_url = url_for('admin', _external=True)
        
        for admin in admin_users:
            if admin.email:
                admin_subject = f"New User Registration: {username} - Action Required"
                admin_body = f"""
Hello {admin.username},

A new user has registered and is awaiting approval:

**Username:** {username}
**Email:** {email}
**Registration Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

**Action Required:**
1. Review the new registration in the Admin Panel: {admin_panel_url}
2. Check the Messages Board for details: {messages_url}
3. Approve or reject the user account as appropriate

---
Tamara Care Log
"""
                admin_html_body = f"""
<html>
<body>
<h2>New User Registration - Action Required</h2>
<p>Hello <strong>{admin.username}</strong>,</p>
<p>A new user has registered and is awaiting approval:</p>
<ul>
<li><strong>Username:</strong> {username}</li>
<li><strong>Email:</strong> {email}</li>
<li><strong>Registration Date:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</li>
</ul>
<p><strong>Action Required:</strong></p>
<ol>
<li>Review the new registration in the <a href="{admin_panel_url}" style="color: #0d6efd;">Admin Panel</a></li>
<li>Check the <a href="{messages_url}" style="color: #0d6efd;">Messages Board</a> for details</li>
<li>Approve or reject the user account as appropriate</li>
</ol>
<hr>
<p><small>Tamara Care Log</small></p>
</body>
</html>
"""
                try:
                    send_email(admin_subject, admin_body, to_email=admin.email, html_body=admin_html_body)
                except Exception as e:
                    print(f"Error sending admin notification email to {admin.email}: {e}")
        
        # region agent log
        try:
            with open(r"a:\apps\nxapps\tamara_care_log\.cursor\debug.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H2",
                    "location": "app.py:register",
                    "message": "register committed",
                    "data": {
                        "username": username,
                        "user_id": new_user.id,
                        "db_uri": app.config.get('SQLALCHEMY_DATABASE_URI'),
                        "user_count": User.query.count()
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # endregion agent log
        flash('Registration successful! Awaiting admin approval. Please check your email for updates.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate secure token
            token = secrets.token_urlsafe(32)
            user.reset_token = token
            user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
            db.session.commit()
            
            # Send password reset email
            reset_url = url_for('reset_password', token=token, _external=True)
            subject = "Password Reset Request - Tamara Care App"
            body = f"""
You requested a password reset for your Tamara Care App account.

Username: {user.username}

Click the link below to reset your password (valid for 1 hour):
{reset_url}

If you didn't request this, please ignore this email. Your password will not be changed.

---
Tamara Care App
"""
            html_body = f"""
<html>
<body>
<h2>Password Reset Request</h2>
<p>You requested a password reset for your Tamara Care App account.</p>
<p><strong>Username:</strong> {user.username}</p>
<p>Click the link below to reset your password (valid for 1 hour):</p>
<p><a href="{reset_url}" style="background-color: #0d6efd; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Reset Password</a></p>
<p>Or copy and paste this URL into your browser:</p>
<p style="word-break: break-all;">{reset_url}</p>
<p><small>If you didn't request this, please ignore this email. Your password will not be changed.</small></p>
<hr>
<p><small>Tamara Care App</small></p>
</body>
</html>
"""
            send_email(subject, body, to_email=user.email, html_body=html_body)
        
        # Always show success message (security: don't reveal if email exists)
        flash('If an account with that email exists, a password reset link has been sent.', 'success')
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token"""
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
        flash('Invalid or expired reset token. Please request a new password reset.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        if password != password_confirm:
            flash('Passwords do not match!', 'error')
            return render_template('reset_password.html', token=token)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'error')
            return render_template('reset_password.html', token=token)
        
        # Update password and clear reset token
        user.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        
        flash('Password reset successful! You can now login with your new password.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/forgot-username', methods=['GET', 'POST'])
def forgot_username():
    """Handle forgot username requests"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Send username reminder email
            subject = "Username Reminder - Tamara Care App"
            body = f"""
You requested a reminder of your username for the Tamara Care App.

Your username is: {user.username}

You can login at: {url_for('login', _external=True)}

If you didn't request this, please ignore this email.

---
Tamara Care App
"""
            html_body = f"""
<html>
<body>
<h2>Username Reminder</h2>
<p>You requested a reminder of your username for the Tamara Care App.</p>
<p><strong>Your username is:</strong> {user.username}</p>
<p>You can <a href="{url_for('login', _external=True)}">login here</a>.</p>
<p><small>If you didn't request this, please ignore this email.</small></p>
<hr>
<p><small>Tamara Care App</small></p>
</body>
</html>
"""
            send_email(subject, body, to_email=user.email, html_body=html_body)
        
        # Always show success message (security: don't reveal if email exists)
        flash('If an account with that email exists, your username has been sent to your email.', 'success')
        return redirect(url_for('login'))
    
    return render_template('forgot_username.html')

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

    # Get calendar events for the same date range
    from datetime import datetime, timedelta, timezone
    calendar_events = []  # Initialize as empty list
    combined_entries = []  # Combined timeline
    
    try:
        # Check if CalendarEvent table exists
        try:
            inspector = db.inspect(db.engine)
            table_exists = inspector.has_table('calendar_events')
        except:
            table_exists = False
        
        if table_exists:
            # SQLite stores datetimes as naive (no timezone), so we need to use naive datetimes for comparison
            # Always show events from 1 week ago to future (matching the sync script logic)
            cal_start_naive = datetime.utcnow() - timedelta(days=7)
            
            if start_date or end_date:
                # If date filters are set, use them but extend the range
                if start_date:
                    try:
                        cal_start_filter = datetime.strptime(start_date, '%Y-%m-%d')
                        # Use the earlier of the two (filter date or 1 week ago)
                        cal_start_naive = min(cal_start_naive, cal_start_filter)
                    except:
                        pass  # Keep default cal_start_naive
                
                if end_date:
                    try:
                        cal_end_naive = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                    except:
                        cal_end_naive = datetime.utcnow() + timedelta(days=365)
                else:
                    cal_end_naive = datetime.utcnow() + timedelta(days=365)
                
                calendar_events = CalendarEvent.query.filter(
                    CalendarEvent.start_datetime >= cal_start_naive,
                    CalendarEvent.start_datetime <= cal_end_naive
                ).order_by(CalendarEvent.start_datetime.asc()).all()
            else:
                # Default: last week to future (no upper limit for future events)
                calendar_events = CalendarEvent.query.filter(
                    CalendarEvent.start_datetime >= cal_start_naive
                ).order_by(CalendarEvent.start_datetime.asc()).all()
    except Exception as e:
        # If there's an error, just show empty list
        import traceback
        print(f"ERROR fetching calendar events: {e}")
        traceback.print_exc()
        calendar_events = []  # Ensure it's always a list
    
    # Create combined timeline: merge logs and calendar events
    # Add log entries
    for log in logs:
        combined_entries.append({
            'type': 'log',
            'datetime': log.activity_datetime,
            'data': log
        })
    
    # Add calendar events - create TWO entries per event (start and end)
    for event in calendar_events:
        # Convert UTC to Melbourne time for sorting
        start_dt = utc_to_melbourne(event.start_datetime) if event.start_datetime else None
        end_dt = utc_to_melbourne(event.end_datetime) if event.end_datetime else None
        
        if start_dt:
            # Add START entry
            combined_entries.append({
                'type': 'calendar_start',
                'datetime': start_dt,
                'data': event
            })
        
        if end_dt:
            # Add END entry
            combined_entries.append({
                'type': 'calendar_end',
                'datetime': end_dt,
                'data': event
            })
    
    # Sort combined entries by datetime (most recent first)
    # If times overlap, prioritize finish entries before start entries
    combined_entries.sort(key=lambda x: (
        x['datetime'], 
        0 if x['type'] == 'calendar_end' else 1 if x['type'] == 'calendar_start' else 2
    ), reverse=True)

    return render_template(
        'view_logs.html',
        logs=logs,
        labels=labels,
        datasets=datasets,
        combined_datasets=combined_datasets,
        carer_names=carer_names,
        selected_carer=carer_name,
        calendar_events=calendar_events,
        combined_entries=combined_entries  # Add combined timeline
    )

@app.route('/admin/tenant-plan', methods=['GET', 'POST'])
def admin_tenant_plan():
    """Admin page to manage tenant subscription plan"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    # Get default tenant (id=1 for single-tenant)
    tenant = Tenant.query.get(1)
    if not tenant:
        # Create default tenant if it doesn't exist
        starter_plan = Plan.query.filter_by(code='starter').first()
        if starter_plan:
            tenant = Tenant(id=1, name='Default Tenant', plan_id=starter_plan.id)
            db.session.add(tenant)
            db.session.commit()
        else:
            flash('Error: No plans found. Please run seed_feature_flags.py first.', 'error')
            return redirect(url_for('admin'))
    
    if request.method == 'POST':
        new_plan_id = request.form.get('plan_id', type=int)
        if new_plan_id:
            plan = Plan.query.get(new_plan_id)
            if plan:
                tenant.plan_id = new_plan_id
                db.session.commit()
                flash(f'Tenant plan updated to {plan.display_name}', 'success')
                return redirect(url_for('admin_tenant_plan'))
            else:
                flash('Invalid plan selected', 'error')
    
    # Get all plans (show all for admin, not just active)
    all_plans = Plan.query.order_by(Plan.id).all()
    
    # Debug: Log plan count
    print(f"[DEBUG] Found {len(all_plans)} plans in database")
    for p in all_plans:
        print(f"  - Plan {p.id}: {p.code} ({p.display_name}), active={p.is_active}")
    
    if not all_plans:
        flash('No plans found in database. Please run seed_feature_flags.py to create plans.', 'warning')
    elif len(all_plans) < 4:
        flash(f'Warning: Only {len(all_plans)} plan(s) found. Expected 4. Please run seed_feature_flags.py.', 'warning')
    
    current_plan = tenant.plan if tenant else None
    
    return render_template('admin_tenant_plan.html',
                         tenant=tenant,
                         current_plan=current_plan,
                         all_plans=all_plans)

# --- Admin-Managed Activities Routes ---

@app.route('/admin/activities', methods=['GET'])
def admin_activities():
    """List all activities"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activities = Activity.query.order_by(Activity.display_order, Activity.name).all()
    
    # Get log counts for each activity
    activities_with_counts = []
    for activity in activities:
        log_count = CareLogEntry.query.filter_by(activity_name=activity.name).count()
        activities_with_counts.append({
            'activity': activity,
            'log_count': log_count
        })
    
    # Debug: Log activity count
    print(f"[DEBUG] Admin activities route: Found {len(activities)} activities")
    if len(activities) > 0:
        print(f"[DEBUG] First few activities: {[a.name for a in activities[:5]]}")
    
    return render_template('admin_activities.html', activities_with_counts=activities_with_counts)


@app.route('/admin/activities/new', methods=['GET', 'POST'])
def admin_activity_new():
    """Create a new activity"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        if not name:
            flash('Activity name is required', 'error')
            return redirect(url_for('admin_activity_new'))
        
        # Check for duplicate
        existing = Activity.query.filter_by(name=name).first()
        if existing:
            flash(f'Activity "{name}" already exists', 'error')
            return redirect(url_for('admin_activity_new'))
        
        activity = Activity(
            name=name,
            display_order=int(request.form.get('display_order', 0) or 0),
            value_type=request.form.get('value_type', 'dropdown'),
            special_duration=request.form.get('special_duration') or None,
            synonyms=request.form.get('synonyms', '').strip() or None,
            tenant_id=1  # Single-tenant: always 1
        )
        db.session.add(activity)
        db.session.commit()
        flash(f'Activity "{name}" created successfully', 'success')
        return redirect(url_for('admin_activity_edit', activity_id=activity.id))
    
    return render_template('admin_activity_edit.html', activity=None)


@app.route('/admin/activities/<int:activity_id>/edit', methods=['GET', 'POST'])
def admin_activity_edit(activity_id):
    """Edit an activity"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity = Activity.query.get_or_404(activity_id)
    
    # Get log count for display (needed for both GET and POST)
    log_count = CareLogEntry.query.filter_by(activity_name=activity.name).count()
    
    if request.method == 'POST':
        # Update activity
        new_name = request.form.get('name', '').strip()
        if not new_name:
            flash('Activity name is required', 'error')
            return redirect(url_for('admin_activity_edit', activity_id=activity_id))
        
        # Check for duplicate (excluding current activity)
        existing = Activity.query.filter_by(name=new_name).first()
        if existing and existing.id != activity_id:
            flash(f'Activity "{new_name}" already exists', 'error')
            return redirect(url_for('admin_activity_edit', activity_id=activity_id))
        
        activity.name = new_name
        activity.display_order = int(request.form.get('display_order', 0) or 0)
        activity.value_type = request.form.get('value_type', 'dropdown')
        activity.special_duration = request.form.get('special_duration') or None
        activity.synonyms = request.form.get('synonyms', '').strip() or None
        activity.is_active = request.form.get('is_active') == 'on'
        db.session.commit()
        flash(f'Activity "{activity.name}" updated successfully', 'success')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    return render_template('admin_activity_edit.html', activity=activity, log_count=log_count)


@app.route('/admin/activities/<int:activity_id>/delete', methods=['POST'])
def admin_activity_delete(activity_id):
    """Delete an activity"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity = Activity.query.get_or_404(activity_id)
    activity_name = activity.name
    
    # Check if there are any log entries using this activity
    log_count = CareLogEntry.query.filter_by(activity_name=activity_name).count()
    if log_count > 0:
        flash(f'❌ Cannot delete activity "{activity_name}" because it has {log_count} log entry/entries. '
              f'Deleting this activity will NOT delete the log entries (they will remain with the activity name). '
              f'Set the activity to "Inactive" instead to hide it from the log form while preserving all data.', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    # Count related data that will be deleted
    values_count = ActivityValue.query.filter_by(activity_id=activity_id).count()
    quick_picks_count = QuickPickButton.query.filter_by(activity_id=activity_id).count()
    
    db.session.delete(activity)
    db.session.commit()
    
    flash(f'✅ Activity "{activity_name}" deleted successfully. '
          f'Also deleted {values_count} value(s) and {quick_picks_count} quick pick button(s).', 'success')
    return redirect(url_for('admin_activities'))


@app.route('/admin/activities/<int:activity_id>/values/new', methods=['POST'])
def admin_activity_value_new(activity_id):
    """Add a new value to an activity"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity = Activity.query.get_or_404(activity_id)
    
    label = request.form.get('label', '').strip()
    value = request.form.get('value', '').strip()
    if not label or not value:
        flash('Label and value are required', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    # Validate value range for graph consistency
    try:
        value_num = float(value)
        if value_num > 5 or (value_num < -5 and value_num >= -10):
            flash(f'⚠️ Warning: Value "{value}" is outside recommended range (1-5 or -5 to -1). This may cause graph scaling issues. Use decimals (1.5, 2.5) for more options.', 'warning')
        elif value_num < -10 or value_num > 100:
            flash(f'⚠️ Warning: Value "{value}" is very large. This will cause graph scaling problems. Consider using a value between 1-5 or negative range -5 to -1.', 'warning')
    except ValueError:
        # Not a number, that's okay (might be text-based)
        pass
    
    activity_value = ActivityValue(
        activity_id=activity_id,
        label=label,
        value=value,
        description=request.form.get('description', '').strip() or None,
        display_order=int(request.form.get('display_order', 0) or 0),
        synonyms=request.form.get('synonyms', '').strip() or None
    )
    db.session.add(activity_value)
    db.session.commit()
    flash(f'Value "{label}" added successfully', 'success')
    return redirect(url_for('admin_activity_edit', activity_id=activity_id))


@app.route('/admin/activities/<int:activity_id>/values/<int:value_id>/edit', methods=['POST'])
def admin_activity_value_edit(activity_id, value_id):
    """Edit an activity value"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity_value = ActivityValue.query.get_or_404(value_id)
    if activity_value.activity_id != activity_id:
        flash('Value does not belong to this activity', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    label = request.form.get('label', '').strip()
    value = request.form.get('value', '').strip()
    if not label or not value:
        flash('Label and value are required', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    # Validate value range for graph consistency
    try:
        value_num = float(value)
        if value_num > 5 or (value_num < -5 and value_num >= -10):
            flash(f'⚠️ Warning: Value "{value}" is outside recommended range (1-5 or -5 to -1). This may cause graph scaling issues. Use decimals (1.5, 2.5) for more options.', 'warning')
        elif value_num < -10 or value_num > 100:
            flash(f'⚠️ Warning: Value "{value}" is very large. This will cause graph scaling problems. Consider using a value between 1-5 or negative range -5 to -1.', 'warning')
    except ValueError:
        # Not a number, that's okay (might be text-based)
        pass
    
    activity_value.label = label
    activity_value.value = value
    activity_value.description = request.form.get('description', '').strip() or None
    activity_value.display_order = int(request.form.get('display_order', 0) or 0)
    activity_value.synonyms = request.form.get('synonyms', '').strip() or None
    db.session.commit()
    flash(f'Value "{label}" updated successfully', 'success')
    return redirect(url_for('admin_activity_edit', activity_id=activity_id))


@app.route('/admin/activities/<int:activity_id>/values/<int:value_id>/delete', methods=['POST'])
def admin_activity_value_delete(activity_id, value_id):
    """Delete an activity value"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity_value = ActivityValue.query.get_or_404(value_id)
    if activity_value.activity_id != activity_id:
        flash('Value does not belong to this activity', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    # Check if there are any log entries using this value
    # We check by matching both value and value_type (label) since logs store both
    activity = activity_value.activity
    log_count = CareLogEntry.query.filter_by(
        activity_name=activity.name
    ).filter(
        (CareLogEntry.value == activity_value.value) |
        (CareLogEntry.value_type == activity_value.label)
    ).count()
    
    if log_count > 0:
        flash(f'Cannot delete value "{activity_value.label}" because it has {log_count} log entry/entries. Consider keeping it for historical data, or update the log entries first.', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    label = activity_value.label
    db.session.delete(activity_value)
    db.session.commit()
    flash(f'Value "{label}" deleted successfully', 'success')
    return redirect(url_for('admin_activity_edit', activity_id=activity_id))


@app.route('/admin/activities/<int:activity_id>/quick-picks/new', methods=['POST'])
def admin_quick_pick_new(activity_id):
    """Add a new quick pick button"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    activity = Activity.query.get_or_404(activity_id)
    
    button_text = request.form.get('button_text', '').strip()
    if not button_text:
        flash('Button text is required', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    activity_value_id = request.form.get('activity_value_id', type=int) or None
    value = request.form.get('value', '').strip() or None
    value_type = request.form.get('value_type', '').strip() or None
    
    # All three are optional - button can just select activity, or pre-fill activity + value
    # If activity_value_id is set, use it. Otherwise use manual value/value_type if provided.
    # If none are set, button will just select the activity (carer picks value on log page)
    
    quick_pick = QuickPickButton(
        activity_id=activity_id,
        activity_value_id=activity_value_id,
        value=value,
        value_type=value_type,
        button_text=button_text,
        display_order=int(request.form.get('display_order', 0) or 0)
    )
    db.session.add(quick_pick)
    db.session.commit()
    flash(f'Quick pick button "{button_text}" added successfully', 'success')
    return redirect(url_for('admin_activity_edit', activity_id=activity_id))


@app.route('/admin/activities/<int:activity_id>/quick-picks/<int:quick_pick_id>/delete', methods=['POST'])
def admin_quick_pick_delete(activity_id, quick_pick_id):
    """Delete a quick pick button"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['carer_id']).first()
    if not user or not user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    quick_pick = QuickPickButton.query.get_or_404(quick_pick_id)
    if quick_pick.activity_id != activity_id:
        flash('Quick pick does not belong to this activity', 'error')
        return redirect(url_for('admin_activity_edit', activity_id=activity_id))
    
    button_text = quick_pick.button_text
    db.session.delete(quick_pick)
    db.session.commit()
    flash(f'Quick pick button "{button_text}" deleted successfully', 'success')
    return redirect(url_for('admin_activity_edit', activity_id=activity_id))


@app.route('/admin')
def admin():
    admin_user = None
    if 'carer_id' in session:
        admin_user = User.query.filter_by(username=session['carer_id']).first()

    if 'carer_id' not in session or not admin_user or not admin_user.is_admin:
        # region agent log
        try:
            with open(r"a:\apps\nxapps\tamara_care_log\.cursor\debug.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H3",
                    "location": "app.py:admin",
                    "message": "admin access blocked",
                    "data": {
                        "session_user": session.get('carer_id'),
                        "db_uri": app.config.get('SQLALCHEMY_DATABASE_URI')
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # endregion agent log
        return redirect(url_for('login'))

    users = User.query.all()
    # region agent log
    try:
        with open(r"a:\apps\nxapps\tamara_care_log\.cursor\debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1",
                "location": "app.py:admin",
                "message": "admin users fetched",
                "data": {
                    "session_user": session.get('carer_id'),
                    "db_uri": app.config.get('SQLALCHEMY_DATABASE_URI'),
                    "user_count": len(users),
                    "sample_usernames": [u.username for u in users[:5]]
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # endregion agent log
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
    
    # Check if user is admin or super_user
    current_user = User.query.filter_by(username=session['carer_id']).first()
    if not current_user or not current_user_is_admin():
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
            # Update email if changed (with validation)
            new_email = request.form.get('email', '').strip().lower()
            if new_email and new_email != user.email:
                # Check if email is already taken by another user
                existing_user = User.query.filter_by(email=new_email).first()
                if existing_user and existing_user.id != user.id:
                    flash(f'Email {new_email} is already in use by another account.', 'error')
                    return redirect(url_for('carer_profile', user_id=user_id))
                user.email = new_email
                flash('Email address updated successfully!', 'success')
            
            # Update role (only super_user can change roles, admin can change to carer/professional)
            if current_user_is_super_user():
                # Super user can change any role
                new_role = request.form.get('role', '').strip()
                if new_role in ('super_user', 'admin', 'professional', 'organization', 'carer', 'readonly'):
                    user.role = new_role
                    # Update is_admin flag for backward compatibility
                    user.is_admin = (new_role in ('super_user', 'admin'))
                    # Update is_readonly flag
                    user.is_readonly = (new_role == 'readonly')
                
                # Update professional_type if role is professional
                if user.role == 'professional':
                    user.professional_type = request.form.get('professional_type', '').strip() or None
                else:
                    user.professional_type = None
                
                # Update organization_id if role is organization
                if user.role == 'organization':
                    org_id = request.form.get('organization_id', '').strip()
                    user.organization_id = int(org_id) if org_id and org_id.isdigit() else None
                else:
                    user.organization_id = None
            elif current_user_is_admin() and not current_user_is_super_user():
                # Regular admin can change to carer or professional (but not super_user or admin)
                new_role = request.form.get('role', '').strip()
                if new_role in ('professional', 'organization', 'carer', 'readonly'):
                    user.role = new_role
                    user.is_admin = False  # Remove admin flag if changing role
                    user.is_readonly = (new_role == 'readonly')
                    
                    if user.role == 'professional':
                        user.professional_type = request.form.get('professional_type', '').strip() or None
                    else:
                        user.professional_type = None
            
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


# -----------------------------
# Focused Tasks (5-stage)
# -----------------------------

@app.route('/focus-tasks', methods=['GET'])
def focus_tasks():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    tasks = FocusTask.query.order_by(
        FocusTask.status.desc(),
        FocusTask.focus_start.desc().nullslast(),
        FocusTask.created_at.desc()
    ).all()
    return render_template('focus_tasks.html', tasks=tasks)


@app.route('/focus-tasks/<int:task_id>', methods=['GET', 'POST'])
def focus_task_detail(task_id):
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    task = FocusTask.query.get_or_404(task_id)
    stages = FocusStage.query.order_by(FocusStage.order).all()

    if request.method == 'POST':
        stage_code = request.form.get('stage_code')
        title = (request.form.get('title') or '').strip()
        detail = (request.form.get('detail') or '').strip()

        if not title or not detail or not stage_code:
            flash('Title, detail, and stage are required.', 'error')
            return redirect(url_for('focus_task_detail', task_id=task_id))

        # Permission: clinicians/admins all stages; carers only CHECK (C)
        if not current_user_is_clinician() and stage_code != 'C':
            abort(403)

        entry = FocusEntry(
            task_id=task.id,
            stage_code=stage_code,
            title=title,
            detail=detail,
            added_by=session.get('carer_id', 'unknown')
        )
        db.session.add(entry)
        db.session.commit()
        flash('Entry added.', 'success')
        return redirect(url_for('focus_task_detail', task_id=task_id))

    entries_by_stage = {s.code: [] for s in stages}
    for e in task.entries:
        entries_by_stage.setdefault(e.stage_code, []).append(e)
    for k in entries_by_stage:
        entries_by_stage[k].sort(key=lambda x: x.entry_date, reverse=True)

    return render_template(
        'focus_task_detail.html',
        task=task,
        stages=stages,
        entries_by_stage=entries_by_stage,
        is_clinician=current_user_is_clinician()
    )


@app.route('/focus-tasks/new', methods=['GET', 'POST'])
def focus_task_new():
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    if not current_user_is_clinician():
        abort(403)

    if request.method == 'POST':
        title = (request.form.get('title') or '').strip()
        short_code = (request.form.get('short_code') or '').strip() or None
        description = request.form.get('description', '').strip()
        status = request.form.get('status', 'active')
        focus_start = parse_date_field(request.form.get('focus_start'))
        focus_end = parse_date_field(request.form.get('focus_end'))
        
        # Get auto-capture activities (multiple checkboxes)
        auto_capture_activities = request.form.getlist('auto_capture_activities')
        import json
        auto_capture_json = json.dumps(auto_capture_activities) if auto_capture_activities else None

        if not title:
            flash('Title is required.', 'error')
            return redirect(url_for('focus_task_new'))

        task = FocusTask(
            title=title,
            short_code=short_code,
            description=description,
            status=status,
            focus_start=focus_start,
            focus_end=focus_end,
            created_by=session.get('carer_id', 'unknown'),
            auto_capture_activities=auto_capture_json
        )
        db.session.add(task)
        db.session.commit()
        flash('Focused task created.', 'success')
        return redirect(url_for('focus_tasks'))

    return render_template('focus_task_new.html')


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

def is_message_recipient(message, user_db_id):
    """Check if a user is a recipient of a message (supports both old single recipient and new multiple recipients)"""
    # Check old single recipient field (backward compatibility)
    if message.recipient_id == user_db_id:
        return True
    # Check new multiple recipients table
    if message.recipients:
        return any(r.recipient_id == user_db_id for r in message.recipients)
    return False

def get_unread_message_count(user_id=None):
    """Get count of unread direct messages for a user (including messages with unread replies)"""
    if not user_id:
        if 'carer_id' not in session:
            return 0
        user_id = session['carer_id']
    
    user = User.query.filter_by(username=user_id).first()
    if not user:
        return 0
    
    user_db_id = user.id
    tenant_id = get_tenant_id_for_user(user_id)
    
    # Get all messages where user is recipient (check both old single recipient and new multiple recipients)
    # Messages where user is in recipient_id (backward compatibility)
    messages_by_recipient_id = DirectMessage.query.filter_by(
        recipient_id=user_db_id,
        tenant_id=tenant_id
    ).all()
    
    # Messages where user is in DirectMessageRecipient table
    recipient_records = DirectMessageRecipient.query.filter_by(recipient_id=user_db_id).all()
    message_ids_from_recipients = {r.message_id for r in recipient_records}
    messages_by_recipient_table = DirectMessage.query.filter(
        DirectMessage.id.in_(message_ids_from_recipients),
        DirectMessage.tenant_id == tenant_id
    ).all() if message_ids_from_recipients else []
    
    # Combine and deduplicate
    all_received_ids = {m.id for m in messages_by_recipient_id} | {m.id for m in messages_by_recipient_table}
    received_messages = DirectMessage.query.filter(
        DirectMessage.id.in_(all_received_ids)
    ).all() if all_received_ids else []
    
    # Get read status for messages
    read_messages = DirectMessageRead.query.filter_by(user_id=user_db_id).all()
    read_message_ids = {rm.message_id for rm in read_messages}
    
    # Count unread messages
    unread_messages = [m for m in received_messages if m.id not in read_message_ids]
    
    # Also check for messages with unread replies (where user is sender)
    sent_messages = DirectMessage.query.filter_by(
        sender_id=user_db_id,
        tenant_id=tenant_id
    ).all()
    
    # Get all read reply IDs for this user
    read_replies = DirectMessageReplyRead.query.filter_by(user_id=user_db_id).all()
    read_reply_ids = {rr.reply_id for rr in read_replies}
    
    # Check sent messages for unread replies
    messages_with_unread_replies = []
    for msg in sent_messages:
        # Get all replies to this message
        replies = DirectMessageReply.query.filter_by(message_id=msg.id).all()
        # Check if any reply is unread by this user
        unread_replies = [r for r in replies if r.id not in read_reply_ids and r.created_by != user_db_id]
        if unread_replies:
            messages_with_unread_replies.append(msg.id)
    
    # Total unread = unread received messages + sent messages with unread replies
    unread_count = len(unread_messages) + len(messages_with_unread_replies)
    return unread_count

def get_unread_swap_request_count():
    """Get count of pending swap requests where current user is target or requester, or all for admins"""
    try:
        if 'carer_id' not in session:
            return 0
        
        user = User.query.filter_by(username=session['carer_id']).first()
        if not user:
            return 0
        
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            return 0
        
        is_admin = current_user_is_admin()
        
        if is_admin:
            # Admins see all pending swap requests in their tenant
            all_pending = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
                Shift.tenant_id == tenant_id,
                ShiftSwapRequest.status == 'pending'
            ).count()
            return all_pending
        else:
            # Regular users see only requests directed to them or open requests
            # Count pending swap requests where user is the target carer
            requests_to_me = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
                Shift.tenant_id == tenant_id,
                ShiftSwapRequest.target_carer_id == user.id,
                ShiftSwapRequest.status == 'pending'
            ).count()
            
            # Count open requests (no target, but not from current user)
            open_requests = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
                Shift.tenant_id == tenant_id,
                ShiftSwapRequest.target_carer_id == None,
                ShiftSwapRequest.status == 'pending',
                ShiftSwapRequest.requester_id != user.id
            ).count()
            
            return requests_to_me + open_requests
    except Exception as e:
        # Return 0 on any error to prevent breaking the header
        print(f"Error getting swap request count: {e}")
        return 0

# Make function available to all templates
@app.context_processor
def inject_notice_count():
    return dict(
        get_unread_notice_count=get_unread_notice_count,
        get_unread_message_count=get_unread_message_count,
        get_unread_swap_request_count=get_unread_swap_request_count
    )

def can_access_feature(feature_name):
    """
    Check if current user can access a feature.
    Returns tuple: (can_access: bool, reason: str or None)
    """
    if 'carer_id' not in session:
        return (False, "Not logged in")
    
    user = User.query.filter_by(username=session.get('carer_id')).first()
    if not user:
        return (False, "User not found")
    
    user_role = (user.role or 'carer').lower()
    tenant_id = get_tenant_id_for_user(user.id)
    
    # Feature-specific checks
    if feature_name == 'add_analysis':
        if user_role in ('super_user', 'admin', 'professional'):
            return (True, None)
        return (False, "Requires Professional tier or higher")
    
    if feature_name == 'focused_tasks':
        # Everyone can view focused tasks, but only Admin/Pro can edit/create
        return (True, None)
    
    if feature_name == 'focused_tasks_edit':
        # Only Admin/Pro can edit or create focused tasks
        if user_role in ('super_user', 'admin', 'professional'):
            return (True, None)
        return (False, "Requires Admin or Professional role to edit focused tasks")
    
    if feature_name == 'shifts':
        if check_feature_enabled(tenant_id, 'booking.view'):
            return (True, None)
        return (False, "Requires Professional tier subscription")
    
    if feature_name == 'manage_services':
        if user_role in ('super_user', 'admin') and check_feature_enabled(tenant_id, 'booking.manage'):
            return (True, None)
        return (False, "Requires Admin role and Professional tier subscription")
    
    if feature_name == 'admin_panel':
        if user_role in ('super_user', 'admin'):
            return (True, None)
        return (False, "Requires Admin or Super User role")
    
    # Default: allow access
    return (True, None)

def get_current_tenant_id():
    """Get current tenant ID for the logged-in user"""
    if 'carer_id' not in session:
        return 1  # Default tenant
    user = User.query.filter_by(username=session.get('carer_id')).first()
    if not user:
        return 1
    return get_tenant_id_for_user(user.id)

@app.context_processor
def inject_role_helpers():
    from models import Tenant
    return dict(
        current_user_is_admin=current_user_is_admin,
        current_user_is_super_user=current_user_is_super_user,
        current_user_is_clinician=current_user_is_clinician,
        current_user_is_professional=current_user_is_professional,
        current_user_is_readonly=current_user_is_readonly,
        current_user_can_edit=current_user_can_edit,
        current_user_role=current_user_role,
        get_accessible_tenants_for_org_user=get_accessible_tenants_for_org_user,
        can_access_feature=can_access_feature,
        get_current_tenant_id=get_current_tenant_id,
        check_feature_enabled=check_feature_enabled,
        Tenant=Tenant,
    )

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

@app.route('/calendar')
def calendar():
    """Tamara's schedule calendar view - accessible to all logged-in users"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    # Get calendar URL from config or URL parameter
    calendar_embed_url = request.args.get('url', '') or app.config.get('TAMARA_CALENDAR_URL', '')
    
    return render_template('admin_calendar.html', calendar_url=calendar_embed_url)

def sync_calendar_from_ical():
    """Sync calendar events from public iCal feed - extracted from sync_calendar_ical.py"""
    try:
        from icalendar import Calendar
    except ImportError:
        raise ImportError("icalendar module is not installed. Run: pip install icalendar")
    
    from datetime import datetime, timedelta, timezone
    import urllib.request
    import urllib.error
    
    # iCal feed URL
    ICAL_URLS = [
        'https://calendar.google.com/calendar/ical/fitzmauricetamara%40gmail.com/private-58c1f313cd7ffa0200b926945cd3faf7/basic.ics'
    ]
    
    try:
        # Try each URL until one works
        ical_data = None
        for url in ICAL_URLS:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    ical_data = response.read()
                    break
            except Exception as e:
                continue
        
        if not ical_data:
            return False
        
        # Parse iCal
        cal = Calendar.from_ical(ical_data)
        
        # Calculate date range: 1 week ago to future
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=7)
        
        new_count = 0
        updated_count = 0
        skipped_count = 0
        
        for component in cal.walk():
            if component.name == "VEVENT":
                event_id = str(component.get('UID', ''))
                if not event_id:
                    continue
                
                dtstart = component.get('DTSTART')
                if not dtstart:
                    continue
                
                # Convert to datetime
                start_dt = dtstart.dt
                if isinstance(start_dt, datetime):
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                else:
                    start_dt = datetime.combine(start_dt, datetime.min.time())
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                
                # Skip events older than 1 week
                if start_dt < cutoff_date and start_dt < now:
                    skipped_count += 1
                    continue
                
                end_dt = None
                dtend = component.get('DTEND')
                if dtend:
                    end_dt = dtend.dt
                    if not isinstance(end_dt, datetime):
                        end_dt = datetime.combine(end_dt, datetime.min.time())
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    elif end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                
                title = str(component.get('SUMMARY', 'No Title'))
                description = str(component.get('DESCRIPTION', ''))
                location = str(component.get('LOCATION', ''))
                
                # Check if event exists
                existing = CalendarEvent.query.filter_by(google_event_id=event_id).first()
                
                event_data = {
                    'title': title,
                    'description': description,
                    'start_datetime': start_dt,
                    'end_datetime': end_dt,
                    'location': location,
                    'updated_at': datetime.utcnow()
                }
                
                if existing:
                    for key, value in event_data.items():
                        setattr(existing, key, value)
                    updated_count += 1
                else:
                    event_data['google_event_id'] = event_id
                    event_data['created_at'] = datetime.utcnow()
                    new_event = CalendarEvent(**event_data)
                    db.session.add(new_event)
                    new_count += 1
        
        # Delete old events no longer in feed
        old_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        old_events = CalendarEvent.query.filter(
            CalendarEvent.start_datetime < old_cutoff
        ).all()
        
        current_event_ids = set()
        for component in cal.walk():
            if component.name == "VEVENT":
                event_id = str(component.get('UID', ''))
                if event_id:
                    current_event_ids.add(event_id)
        
        deleted_count = 0
        for old_event in old_events:
            if old_event.google_event_id not in current_event_ids:
                db.session.delete(old_event)
                deleted_count += 1
        
        db.session.commit()
        return True
        
    except Exception as e:
        print(f"Error syncing calendar: {str(e)}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return False

@app.route('/calendar-events/sync', methods=['POST'])
def sync_calendar_events():
    """Manually trigger calendar sync from Google Calendar"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    try:
        success = sync_calendar_from_ical()
        
        if success:
            flash('Calendar sync completed successfully! The page will refresh to show updated data.', 'success')
        else:
            flash('Calendar sync completed with errors. Check server logs for details.', 'warning')
        
    except Exception as e:
        flash(f'Error syncing calendar: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
    
    return redirect(url_for('calendar_events'))

@app.route('/calendar-events')
def calendar_events():
    """View synced calendar events from database with date range filtering"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    from datetime import datetime, timedelta, timezone
    
    # Safely check if CalendarEvent table exists
    try:
        # Try to query - if table doesn't exist, this will fail
        CalendarEvent.query.limit(1).all()
    except Exception as e:
        # Table doesn't exist - return empty results
        return render_template('calendar_events.html',
            events=[],
            total_events=0,
            total_in_db=0,
            last_sync=None,
            start_date=request.args.get('start_date', ''),
            end_date=request.args.get('end_date', ''),
            date_range_info=None
        )
    
    # Get date range filters from query parameters
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')
    
    try:
        # Build query
        query = CalendarEvent.query
        
        # Parse and apply date filters (using naive datetime for SQLite)
        if start_date_str:
            try:
                start_date_naive = datetime.strptime(start_date_str, '%Y-%m-%d')
                query = query.filter(CalendarEvent.start_datetime >= start_date_naive)
            except ValueError:
                pass  # Invalid date format, ignore
        
        if end_date_str:
            try:
                end_date_naive = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)
                query = query.filter(CalendarEvent.start_datetime <= end_date_naive)
            except ValueError:
                pass  # Invalid date format, ignore
        
        # If no filters, default to last week to future
        if not start_date_str and not end_date_str:
            cal_start_naive = datetime.utcnow() - timedelta(days=7)
            query = query.filter(CalendarEvent.start_datetime >= cal_start_naive)
        
        # Get all events matching the filter
        events = query.order_by(CalendarEvent.start_datetime.asc()).all()
        
        # Get total count in database (for reference)
        total_in_db = CalendarEvent.query.count()
        
        # Get last sync time (most recent updated_at)
        last_sync = None
        last_event = CalendarEvent.query.order_by(CalendarEvent.updated_at.desc()).first()
        if last_event:
            last_sync = last_event.updated_at
        
        # Get date range of all events in DB for reference
        earliest_event = CalendarEvent.query.order_by(CalendarEvent.start_datetime.asc()).first()
        latest_event = CalendarEvent.query.order_by(CalendarEvent.start_datetime.desc()).first()
        
        date_range_info = None
        if earliest_event and latest_event:
            date_range_info = {
                'earliest': earliest_event.start_datetime.date(),
                'latest': latest_event.start_datetime.date()
            }
        
        return render_template('calendar_events.html',
            events=events,
            total_events=len(events),
            total_in_db=total_in_db,
            last_sync=last_sync,
            start_date=start_date_str,
            end_date=end_date_str,
            date_range_info=date_range_info
        )
    except Exception as e:
        # If there's any error, show empty results with error message
        import traceback
        print(f"ERROR in calendar_events route: {e}")
        traceback.print_exc()
        flash(f'Error loading calendar events: {str(e)}', 'danger')
        return render_template('calendar_events.html',
            events=[],
            total_events=0,
            total_in_db=0,
            last_sync=None,
            start_date=start_date_str,
            end_date=end_date_str,
            date_range_info=None
        )

@app.route('/admin/insurance', methods=['GET', 'POST'])
def admin_insurance():
    """Admin view: manage insurance records for all users."""
    if 'carer_id' not in session or not current_user_is_admin():
        return redirect(url_for('login'))

    users = User.query.order_by(User.username.asc()).all()
    records = {r.user_id: r for r in InsuranceRecord.query.all()}

    if request.method == 'POST':
        try:
            user_id = int(request.form.get('user_id', 0))
        except (TypeError, ValueError):
            user_id = 0

        target_user = User.query.get(user_id)
        if not target_user:
            flash('User not found for insurance update.', 'danger')
            return redirect(url_for('admin_insurance'))

        rec = records.get(user_id) or InsuranceRecord(user_id=user_id)

        rec.public_liability_policy = request.form.get('pl_policy') or None
        rec.public_liability_insurer = request.form.get('pl_insurer') or None
        rec.public_liability_coverage = request.form.get('pl_coverage') or None
        rec.public_liability_expiry = parse_date_field(request.form.get('pl_expiry'))

        rec.professional_indemnity_policy = request.form.get('pi_policy') or None
        rec.professional_indemnity_insurer = request.form.get('pi_insurer') or None
        rec.professional_indemnity_coverage = request.form.get('pi_coverage') or None
        rec.professional_indemnity_expiry = parse_date_field(request.form.get('pi_expiry'))

        rec.workers_comp_policy = request.form.get('wc_policy') or None
        rec.workers_comp_insurer = request.form.get('wc_insurer') or None
        rec.workers_comp_expiry = parse_date_field(request.form.get('wc_expiry'))

        rec.car_insurance_policy = request.form.get('car_policy') or None
        rec.car_insurance_insurer = request.form.get('car_insurer') or None
        rec.car_insurance_expiry = parse_date_field(request.form.get('car_expiry'))

        db.session.add(rec)
        db.session.commit()
        flash('Insurance updated.', 'success')
        return redirect(url_for('admin_insurance'))

    return render_template('admin_insurance.html', users=users, records=records)

@app.route('/me/insurance', methods=['GET', 'POST'])
def my_insurance():
    """Self-service view: user can see/update their own insurance."""
    if 'carer_id' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['carer_id']).first_or_404()
    rec = InsuranceRecord.query.filter_by(user_id=user.id).first()

    if request.method == 'POST':
        if not rec:
            rec = InsuranceRecord(user_id=user.id)

        rec.public_liability_policy = request.form.get('pl_policy') or None
        rec.public_liability_insurer = request.form.get('pl_insurer') or None
        rec.public_liability_coverage = request.form.get('pl_coverage') or None
        rec.public_liability_expiry = parse_date_field(request.form.get('pl_expiry'))

        rec.professional_indemnity_policy = request.form.get('pi_policy') or None
        rec.professional_indemnity_insurer = request.form.get('pi_insurer') or None
        rec.professional_indemnity_coverage = request.form.get('pi_coverage') or None
        rec.professional_indemnity_expiry = parse_date_field(request.form.get('pi_expiry'))

        rec.workers_comp_policy = request.form.get('wc_policy') or None
        rec.workers_comp_insurer = request.form.get('wc_insurer') or None
        rec.workers_comp_expiry = parse_date_field(request.form.get('wc_expiry'))

        rec.car_insurance_policy = request.form.get('car_policy') or None
        rec.car_insurance_insurer = request.form.get('car_insurer') or None
        rec.car_insurance_expiry = parse_date_field(request.form.get('car_expiry'))

        db.session.add(rec)
        db.session.commit()
        flash('Your insurance info was saved.', 'success')
        return redirect(url_for('my_insurance'))

    return render_template('my_insurance.html', record=rec)

@app.route('/admin/system-brain')
def admin_system_brain():
    """Admin interface to view/search SYSTEM-BRAIN.sqlite"""
    if 'carer_id' not in session or not current_user_is_admin():
        return redirect(url_for('login'))
    
    import sqlite3
    
    # Path to brain database - A:\ is really C:\ on server, but Flask handles A:\ fine
    brain_db = r"A:\brain\SYSTEM-BRAIN.sqlite"
    
    # If A:\ doesn't exist, try C:\brain\SYSTEM-BRAIN.sqlite
    if not os.path.exists(brain_db):
        brain_db = r"C:\brain\SYSTEM-BRAIN.sqlite"
    
    search_term = request.args.get('search', '').strip()
    
    try:
        conn = sqlite3.connect(brain_db)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        if search_term:
            # Search across description, tags, project, git commit
            c.execute("""
                SELECT date, project, description, files, tags, git_commit_hash, git_branch 
                FROM changes 
                WHERE description LIKE ? OR tags LIKE ? OR project LIKE ? OR git_commit_hash LIKE ?
                ORDER BY date DESC 
                LIMIT 100
            """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
        else:
            # Show last 50 changes
            c.execute("""
                SELECT date, project, description, files, tags, git_commit_hash, git_branch 
                FROM changes 
                ORDER BY date DESC 
                LIMIT 50
            """)
        
        changes = [dict(row) for row in c.fetchall()]
        
        # Get project stats
        c.execute("SELECT COUNT(DISTINCT project) as project_count, COUNT(*) as total_changes FROM changes")
        stats = dict(c.fetchone())
        
        conn.close()
        
    except Exception as e:
        flash(f'Error reading brain database: {str(e)}', 'danger')
        changes = []
        stats = {'project_count': 0, 'total_changes': 0}
    
    return render_template('admin_brain.html', changes=changes, search_term=search_term, stats=stats)

@app.route('/admin/system-brain/projects')
def admin_system_brain_projects():
    """Show all projects and markdown files"""
    if 'carer_id' not in session or not current_user_is_admin():
        return redirect(url_for('login'))
    
    import sqlite3
    
    brain_db = r"A:\brain\SYSTEM-BRAIN.sqlite"
    if not os.path.exists(brain_db):
        brain_db = r"C:\brain\SYSTEM-BRAIN.sqlite"
    
    try:
        conn = sqlite3.connect(brain_db)
        conn.row_factory = sqlite3.Row
        
        # Get all projects
        projects = conn.execute("""
            SELECT name, path, project_type, has_project_knowledge, has_readme, has_mkdocs, last_scan
            FROM projects 
            ORDER BY name
        """).fetchall()
        
        # Get markdown file counts by project
        md_counts = conn.execute("""
            SELECT project_path, file_type, COUNT(*) as count
            FROM markdown_files
            GROUP BY project_path, file_type
        """).fetchall()
        
        # Organize md counts by project
        md_by_project = {}
        for row in md_counts:
            proj_path = row['project_path']
            if proj_path not in md_by_project:
                md_by_project[proj_path] = {}
            md_by_project[proj_path][row['file_type']] = row['count']
        
        projects_list = [dict(p) for p in projects]
        for proj in projects_list:
            proj['md_files'] = md_by_project.get(proj['path'], {})
        
        conn.close()
        
    except Exception as e:
        flash(f'Error reading projects: {str(e)}', 'danger')
        projects_list = []
    
    return render_template('admin_brain_projects.html', projects=projects_list)

@app.route('/admin/system-brain/log-commit', methods=['POST'])
def log_git_commit():
    """API endpoint to log a git commit to System Brain"""
    if 'carer_id' not in session or not current_user_is_admin():
        return jsonify({'error': 'Unauthorized'}), 401
    
    import sqlite3
    import subprocess
    
    data = request.json
    description = data.get('description', '').strip()
    files = data.get('files')
    tags = data.get('tags')
    project_path = data.get('project_path', os.getcwd())
    
    if not description:
        return jsonify({'error': 'Description required'}), 400
    
    project = os.path.basename(project_path.rstrip("\\/"))
    
    # Get git info
    commit_hash = branch = None
    if os.path.exists(os.path.join(project_path, '.git')):
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            commit_hash = result.stdout.strip()[:8] if result.returncode == 0 else None
            
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            branch = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            pass
    
    brain_db = r"A:\brain\SYSTEM-BRAIN.sqlite"
    if not os.path.exists(brain_db):
        brain_db = r"C:\brain\SYSTEM-BRAIN.sqlite"
    
    try:
        conn = sqlite3.connect(brain_db)
        conn.execute("""
            INSERT INTO changes (project, description, files, tags, git_commit_hash, git_branch, git_repo_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (project, description, files, tags, commit_hash, branch, project_path))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Logged: {description}',
            'commit_hash': commit_hash,
            'branch': branch
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== NEW ROUTES FOR LAUNCH READINESS (Added after existing routes) ==========
# These routes are additive only - they don't affect existing functionality
# Can be safely removed if needed

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancers"""
    try:
        # Quick database connectivity check
        db.session.execute(text('SELECT 1'))
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'
    
    return jsonify({
        'status': 'healthy' if db_status == 'connected' else 'degraded',
        'database': db_status,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200 if db_status == 'connected' else 503

@app.route('/legal/terms')
def terms_of_service():
    """Terms of Service page"""
    return render_template('legal/terms.html')

@app.route('/legal/privacy')
def privacy_policy():
    """Privacy Policy page"""
    return render_template('legal/privacy.html')

@app.route('/support')
def support():
    """Support page"""
    return render_template('support/index.html')

@app.route('/support/contact', methods=['GET', 'POST'])
def contact_support():
    """Contact support form"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()
        
        if name and email and message:
            # Send email notification
            subject = f"Support Request from {name} - Tamara Care App"
            body = f"""
Support Request from Tamara Care App

From: {name}
Email: {email}

Message:
{message}

---
This message was sent from the contact form at https://fitzmaurice.net.au/support/contact
"""
            html_body = f"""
<html>
<body>
<h2>Support Request from Tamara Care App</h2>
<p><strong>From:</strong> {name}<br>
<strong>Email:</strong> <a href="mailto:{email}">{email}</a></p>
<h3>Message:</h3>
<p>{message.replace(chr(10), '<br>')}</p>
<hr>
<p><small>This message was sent from the contact form at <a href="https://fitzmaurice.net.au/support/contact">https://fitzmaurice.net.au/support/contact</a></small></p>
</body>
</html>
"""
            # Send to both email addresses
            send_email(subject, body, to_email='tomf@wwave.com.au', html_body=html_body)
            send_email(subject, body, to_email='fitzmauricetamara@gmail.com', html_body=html_body)
            
            flash('Thank you for your message. We will get back to you soon!', 'success')
            return redirect(url_for('support'))
        else:
            flash('Please fill in all fields.', 'error')
    
    return render_template('support/contact.html')

@app.route('/quick/aac', methods=['GET', 'POST'])
def quick_aac():
    """Quick AAC check page - fast logging for carers"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        data = request.get_json(force=True)
        rating = data.get('rating', '')
        focus_words = data.get('focus', [])
        contexts = data.get('contexts', [])
        barriers = data.get('barriers', [])
        note = data.get('note', '').strip()
        
        # Create a log entry for this quick check
        activity_dt = datetime.now()
        log = CareLogEntry(
            carer_id=session['carer_id'],
            carer_name=session['carer_name'],
            activity_name='Communication Level',
            value=str(len(focus_words)) if focus_words else '0',
            value_type=f"AAC Quick Check: {', '.join(focus_words) if focus_words else 'None modelled'}",
            notes=f"Rating: {rating}. Contexts: {', '.join(contexts) if contexts else 'None'}. Barriers: {', '.join(barriers) if barriers else 'None'}. {note}".strip(),
            activity_datetime=activity_dt,
            activity_type='quick_aac'
        )
        db.session.add(log)
        db.session.commit()
        
        # Note: Quick AAC checks are NOT automatically linked to Focused Tasks
        # Carers can manually link entries via the regular log entry form if needed
        
        return jsonify({'ok': True, 'message': 'Saved successfully'})
    
    return render_template('quick_aac.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve images from images folder"""
    return send_from_directory('images', filename)

@app.route('/quick/tamara', methods=['GET', 'POST'])
def quick_tamara():
    """Ultra-simple AAC page for Tamara to use independently"""
    if request.method == 'POST':
        data = request.get_json(force=True)
        action = data.get('action', '')
        label = data.get('label', '')
        
        # Create a log entry for Tamara's communication
        activity_dt = datetime.now()
        # Use a system user or anonymous identifier
        carer_id = session.get('carer_id', 'tamara_self')
        carer_name = session.get('carer_name', 'Tamara (self)')
        
        log = CareLogEntry(
            carer_id=carer_id,
            carer_name=carer_name,
            activity_name='Communication Level',
            value='1',
            value_type=f"Tamara initiated: {label}",
            notes=f"Tamara used Quick AAC page to communicate: {label}",
            activity_datetime=activity_dt,
            activity_type='tamara_quick'
        )
        db.session.add(log)
        db.session.commit()
        
        # Note: Tamara's Quick AAC entries are NOT automatically linked to Focused Tasks
        # Carers can manually link entries via the regular log entry form if needed
        # Email notifications for Tamara's communications are disabled
        # Email is only used for support form submissions and other important notifications
        
        return jsonify({'ok': True, 'message': 'Communication logged'})
    
    return render_template('quick_tamara.html')

# Error handlers (optional - app works without these)
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('errors/403.html'), 403

# ============================================================================
# Direct Messaging System Routes
# ============================================================================

@app.route('/messages')
def messages():
    """View all direct messages (inbox)"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    tenant_id = get_tenant_id_for_user(user_id)
    
    # Get all messages where user is recipient (check both old single recipient and new multiple recipients)
    # Messages where user is in recipient_id (backward compatibility)
    messages_by_recipient_id = DirectMessage.query.filter_by(
        recipient_id=user_db_id,
        tenant_id=tenant_id
    ).all()
    
    # Messages where user is in DirectMessageRecipient table
    recipient_records = DirectMessageRecipient.query.filter_by(recipient_id=user_db_id).all()
    message_ids_from_recipients = {r.message_id for r in recipient_records}
    messages_by_recipient_table = DirectMessage.query.filter(
        DirectMessage.id.in_(message_ids_from_recipients),
        DirectMessage.tenant_id == tenant_id
    ).all() if message_ids_from_recipients else []
    
    # Combine and deduplicate
    all_received_ids = {m.id for m in messages_by_recipient_id} | {m.id for m in messages_by_recipient_table}
    received_messages = DirectMessage.query.filter(
        DirectMessage.id.in_(all_received_ids)
    ).order_by(DirectMessage.created_at.desc()).all() if all_received_ids else []
    
    # Get all messages where user is sender (sent folder)
    sent_messages = DirectMessage.query.filter_by(
        sender_id=user_db_id,
        tenant_id=tenant_id
    ).order_by(DirectMessage.created_at.desc()).all()
    
    # Get read status for messages
    read_messages = DirectMessageRead.query.filter_by(user_id=user_db_id).all()
    read_message_ids = {rm.message_id for rm in read_messages}
    
    # Get read status for replies
    read_replies = DirectMessageReplyRead.query.filter_by(user_id=user_db_id).all()
    read_reply_ids = {rr.reply_id for rr in read_replies}
    
    # Separate received messages into unread and read
    unread_messages = [m for m in received_messages if m.id not in read_message_ids]
    read_messages_list = [m for m in received_messages if m.id in read_message_ids]
    
    # Also include sent messages with unread replies in the unread section
    messages_with_unread_replies = []
    for msg in sent_messages:
        # Get all replies to this message
        replies = DirectMessageReply.query.filter_by(message_id=msg.id).all()
        # Check if any reply is unread by this user (replies from others)
        unread_replies = [r for r in replies if r.id not in read_reply_ids and r.created_by != user_db_id]
        if unread_replies:
            messages_with_unread_replies.append(msg)
    
    # Combine unread received messages with sent messages that have unread replies
    all_unread_messages = unread_messages + messages_with_unread_replies
    
    # All messages (for "All" tab) - combine received and sent
    all_messages_combined = (received_messages + sent_messages)
    # Remove duplicates and sort by date
    seen_ids = set()
    unique_all_messages = []
    for msg in sorted(all_messages_combined, key=lambda x: x.created_at, reverse=True):
        if msg.id not in seen_ids:
            seen_ids.add(msg.id)
            unique_all_messages.append(msg)
    
    # Admin can see all messages in tenant (separate view)
    admin_all_messages = None
    if user.is_admin:
        admin_all_messages = DirectMessage.query.filter_by(
            tenant_id=tenant_id
        ).order_by(DirectMessage.created_at.desc()).limit(50).all()
    
    return render_template('messages.html',
                         received_messages=received_messages,
                         sent_messages=sent_messages,
                         unread_messages=all_unread_messages,
                         read_messages=read_messages_list,
                         read_message_ids=read_message_ids,
                         all_messages=unique_all_messages,
                         admin_all_messages=admin_all_messages,
                         user_id=user_id,
                         user_db_id=user_db_id,
                         is_admin=user.is_admin if user else False,
                         read_reply_ids=read_reply_ids)

@app.route('/messages/new', methods=['GET', 'POST'])
def new_message():
    """Compose a new direct message"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    tenant_id = get_tenant_id_for_user(user_id)
    
    # Check feature flag for messaging
    if not check_feature_enabled(tenant_id, 'messaging.family'):
        flash('Direct messaging is not available on your plan.', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Support both single recipient (backward compatible) and multiple recipients
        recipient_usernames = request.form.getlist('recipient_username')  # Get list of selected recipients
        if not recipient_usernames:
            # Fallback to single recipient field for backward compatibility
            single_recipient = request.form.get('recipient_username', '').strip()
            if single_recipient:
                recipient_usernames = [single_recipient]
        
        subject = request.form.get('subject', '').strip()
        content = request.form.get('content', '').strip()
        message_type = request.form.get('message_type', 'general')
        is_urgent = request.form.get('is_urgent') == 'on'
        related_shift_id = request.form.get('related_shift_id', type=int) or None
        
        if not recipient_usernames or not subject or not content:
            flash('At least one recipient, subject, and content are required!', 'error')
            return redirect(url_for('new_message'))
        
        # Find all recipients
        recipients = []
        for username in recipient_usernames:
            if username:  # Skip empty strings
                recipient = User.query.filter_by(username=username.strip()).first()
                if recipient:
                    recipients.append(recipient)
        
        if not recipients:
            flash('No valid recipients found!', 'error')
            return redirect(url_for('new_message'))
        
        # Check if user can message these recipients based on roles and feature flags
        for recipient in recipients:
            recipient_role = recipient.role if recipient.role else 'carer'
            
            # Check clinician messaging (Pro+ only)
            if recipient_role == 'clinician' and not check_feature_enabled(tenant_id, 'messaging.clinician'):
                flash(f'Messaging clinicians is not available on your plan. Upgrade to Professional tier. (Cannot message {recipient.username})', 'error')
                return redirect(url_for('new_message'))
        
        # Create message (single message for all recipients)
        message = DirectMessage(
            subject=subject,
            content=content,
            sender_id=user_db_id,
            recipient_id=recipients[0].id,  # Set first recipient for backward compatibility
            tenant_id=tenant_id,  # Always set tenant_id
            message_type=message_type,
            is_urgent=is_urgent,
            related_shift_id=related_shift_id
        )
        db.session.add(message)
        db.session.flush()  # Get message ID
        
        # Add all recipients to the message
        for recipient in recipients:
            msg_recipient = DirectMessageRecipient(
                message_id=message.id,
                recipient_id=recipient.id
            )
            db.session.add(msg_recipient)
        
        # Mark as read by sender (they've already seen it)
        message_read = DirectMessageRead(message_id=message.id, user_id=user_db_id)
        db.session.add(message_read)
        db.session.commit()
        
        # Send email notification to all recipients if enabled
        if check_feature_enabled(tenant_id, 'messaging.email_notify'):
            recipient_list = ', '.join([r.username for r in recipients])
            email_subject = f"New Message from {user.username}"
            if is_urgent:
                email_subject = f"URGENT: {email_subject}"
            
            for recipient in recipients:
                recipient_email = recipient.email
                if recipient_email:
                    email_body = f"""
Hello {recipient.username or recipient.full_name or 'there'},

You have received a new direct message in the Tamara Care App.

From: {user.username}
To: {recipient_list}
Subject: {subject}

{content}

Please log in to the app to view and reply to this message.

https://fitzmaurice.net.au/messages/{message.id}

Best regards,
Tamara Care App
"""
                    send_email(email_subject, email_body, to_email=recipient_email)
        
        if len(recipients) > 1:
            flash(f'Message sent successfully to {len(recipients)} recipients!', 'success')
        else:
            flash('Message sent successfully!', 'success')
        return redirect(url_for('messages'))
    
    # GET request - show compose form
    # Get list of users for recipient dropdown
    all_users = User.query.filter_by(is_approved=True).order_by(User.username).all()
    
    return render_template('new_message.html', 
                         all_users=all_users,
                         user_id=user_id)

@app.route('/messages/<int:message_id>')
def view_message(message_id):
    """View a specific message and its replies"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    tenant_id = get_tenant_id_for_user(user_id)
    
    message = DirectMessage.query.get_or_404(message_id)
    
    # Check if user has access (sender, recipient, or admin in same tenant)
    if message.tenant_id != tenant_id:
        flash('You do not have permission to view this message.', 'error')
        return redirect(url_for('messages'))
    
    is_recipient = is_message_recipient(message, user_db_id)
    if message.sender_id != user_db_id and not is_recipient and not (user.is_admin if user else False):
        flash('You do not have permission to view this message.', 'error')
        return redirect(url_for('messages'))
    
    # Don't auto-mark as read - let user manually mark it when they're ready
    # This prevents messages from disappearing if user views but doesn't reply
    
    # Don't auto-mark replies as read - let user manually mark them when ready
    
    # Get read status for message
    read_messages = DirectMessageRead.query.filter_by(user_id=user_db_id).all()
    read_message_ids = {rm.message_id for rm in read_messages}
    
    # Get read reply IDs for template
    read_replies = DirectMessageReplyRead.query.filter_by(user_id=user_db_id).all()
    read_reply_ids = {rr.reply_id for rr in read_replies}
    
    return render_template('view_message.html',
                         message=message,
                         user_id=user_id,
                         user_db_id=user_db_id,
                         is_admin=user.is_admin if user else False,
                         is_recipient=is_recipient,
                         read_reply_ids=read_reply_ids,
                         read_message_ids=read_message_ids)

@app.route('/messages/<int:message_id>/reply', methods=['POST'])
def reply_to_message(message_id):
    """Reply to a direct message"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    tenant_id = get_tenant_id_for_user(user_id)
    
    message = DirectMessage.query.get_or_404(message_id)
    
    # Check tenant access
    if message.tenant_id != tenant_id:
        flash('You do not have permission to reply to this message.', 'error')
        return redirect(url_for('messages'))
    
    # Check if user has access
    is_recipient = is_message_recipient(message, user_db_id)
    if message.sender_id != user_db_id and not is_recipient and not (user.is_admin if user else False):
        flash('You do not have permission to reply to this message.', 'error')
        return redirect(url_for('messages'))
    
    content = request.form.get('reply_content', '').strip()
    
    if content:
        reply = DirectMessageReply(
            message_id=message_id,
            content=content,
            created_by=user_db_id
        )
        db.session.add(reply)
        
        # Notify the other party (sender if recipient replied, recipient if sender replied)
        other_user_id = message.sender_id if message.recipient_id == user_db_id else message.recipient_id
        other_user = User.query.get(other_user_id)
        
        # Mark message as unread for the other party when a reply is added
        # This ensures they get a notification badge
        existing_read = DirectMessageRead.query.filter_by(
            message_id=message_id,
            user_id=other_user_id
        ).first()
        
        if existing_read:
            # Remove the read status so message shows as unread again
            db.session.delete(existing_read)
        
        # Send email notification if enabled
        if check_feature_enabled(tenant_id, 'messaging.email_notify') and other_user and other_user.email:
            email_subject = f"Reply to: {message.subject}"
            email_body = f"""
Hello {other_user.username or other_user.full_name or 'there'},

{user.username} has replied to your message:

{content}

Please log in to the app to view the full conversation.

https://fitzmaurice.net.au/messages/{message_id}

Best regards,
Tamara Care App
"""
            send_email(email_subject, email_body, to_email=other_user.email)
        
        db.session.commit()
        flash('Reply sent successfully!', 'success')
    else:
        flash('Reply content cannot be empty!', 'error')
    
    return redirect(url_for('view_message', message_id=message_id))


@app.route('/messages/<int:message_id>/mark-unread', methods=['POST'])
def mark_message_unread(message_id):
    """Mark a message as unread"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('messages'))
    
    message = DirectMessage.query.get_or_404(message_id)
    
    # Check if user has access
    tenant_id = get_tenant_id_for_user(user_id)
    if message.tenant_id != tenant_id:
        flash('You do not have permission to mark this message as unread.', 'error')
        return redirect(url_for('messages'))
    
    is_recipient = is_message_recipient(message, user_db_id)
    if message.sender_id != user_db_id and not is_recipient and not (user.is_admin if user else False):
        flash('You do not have permission to mark this message as unread.', 'error')
        return redirect(url_for('messages'))
    
    # Remove the read record
    existing_read = DirectMessageRead.query.filter_by(message_id=message_id, user_id=user_db_id).first()
    if existing_read:
        db.session.delete(existing_read)
        db.session.commit()
        flash('Message marked as unread', 'success')
    
    return redirect(url_for('view_message', message_id=message_id))

@app.route('/messages/<int:message_id>/mark-read', methods=['POST'])
def mark_message_read_manual(message_id):
    """Manually mark a message as read"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('messages'))
    
    message = DirectMessage.query.get_or_404(message_id)
    
    # Check if user has access
    tenant_id = get_tenant_id_for_user(user_id)
    if message.tenant_id != tenant_id:
        flash('You do not have permission to mark this message as read.', 'error')
        return redirect(url_for('messages'))
    
    is_recipient = is_message_recipient(message, user_db_id)
    if message.sender_id != user_db_id and not is_recipient and not (user.is_admin if user else False):
        flash('You do not have permission to mark this message as read.', 'error')
        return redirect(url_for('messages'))
    
    # Mark as read
    existing_read = DirectMessageRead.query.filter_by(message_id=message_id, user_id=user_db_id).first()
    if not existing_read:
        message_read = DirectMessageRead(message_id=message_id, user_id=user_db_id)
        db.session.add(message_read)
        db.session.commit()
        flash('Message marked as read', 'success')
    
    return redirect(url_for('view_message', message_id=message_id))

@app.route('/messages/replies/<int:reply_id>/mark-read', methods=['POST'])
def mark_reply_read(reply_id):
    """Mark a reply as read"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['carer_id']
    user = User.query.filter_by(username=user_id).first()
    user_db_id = user.id if user else None
    
    if not user_db_id:
        flash('User not found', 'error')
        return redirect(url_for('messages'))
    
    reply = DirectMessageReply.query.get_or_404(reply_id)
    message = DirectMessage.query.get_or_404(reply.message_id)
    
    # Check if user has access to this message
    tenant_id = get_tenant_id_for_user(user_id)
    if message.tenant_id != tenant_id:
        flash('You do not have permission to mark this reply as read.', 'error')
        return redirect(url_for('messages'))
    
    if message.sender_id != user_db_id and message.recipient_id != user_db_id and not (user.is_admin if user else False):
        flash('You do not have permission to mark this reply as read.', 'error')
        return redirect(url_for('messages'))
    
    # Mark reply as read
    existing_read = DirectMessageReplyRead.query.filter_by(reply_id=reply_id, user_id=user_db_id).first()
    if not existing_read:
        reply_read = DirectMessageReplyRead(reply_id=reply_id, user_id=user_db_id)
        db.session.add(reply_read)
        db.session.commit()
        flash('Reply marked as read', 'success')
    
    return redirect(url_for('view_message', message_id=reply.message_id))

@app.route('/roles-permissions')
def roles_permissions():
    """Display roles and permissions information page"""
    if 'carer_id' not in session:
        flash('Please log in to view roles and permissions', 'info')
        return redirect(url_for('login'))
    return render_template('roles_permissions.html')


# ========== TENANT ACCESS MANAGEMENT ROUTES ==========

@app.route('/admin/tenant-access')
def tenant_access_management():
    """Tenant admin: View/manage who has access to their tenant"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccess, TenantInvitation, TenantAccessRequest
    
    tenant_id = get_current_tenant_id()
    
    # Get active access grants
    active_access = TenantAccess.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).order_by(TenantAccess.granted_at.desc()).all()
    
    # Get pending requests
    pending_requests = TenantAccessRequest.query.filter_by(
        tenant_id=tenant_id,
        status='pending'
    ).order_by(TenantAccessRequest.requested_at.desc()).all()
    
    # Get active invitations
    active_invitations = TenantInvitation.query.filter_by(
        tenant_id=tenant_id,
        is_active=True
    ).order_by(TenantInvitation.created_at.desc()).all()
    
    # Get tenant info
    tenant = Tenant.query.get(tenant_id)
    
    return render_template('admin/tenant_access.html',
                         active_access=active_access,
                         pending_requests=pending_requests,
                         active_invitations=active_invitations,
                         tenant=tenant)


@app.route('/admin/tenant-access/generate-invitation', methods=['POST'])
def generate_tenant_invitation():
    """Generate invitation link/code"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    from models import TenantInvitation
    import secrets
    import string
    
    tenant_id = get_current_tenant_id()
    notes = request.form.get('notes', '').strip()
    
    # Generate unique invitation code (e.g., "TAMARA-ABC123")
    tenant = Tenant.query.get(tenant_id)
    tenant_prefix = tenant.name.upper().replace(' ', '')[:6] if tenant else 'TENANT'
    
    # Generate random suffix
    alphabet = string.ascii_uppercase + string.digits
    suffix = ''.join(secrets.choice(alphabet) for _ in range(6))
    invitation_code = f"{tenant_prefix}-{suffix}"
    
    # Ensure uniqueness
    while TenantInvitation.query.filter_by(invitation_code=invitation_code).first():
        suffix = ''.join(secrets.choice(alphabet) for _ in range(6))
        invitation_code = f"{tenant_prefix}-{suffix}"
    
    # Create invitation
    invitation = TenantInvitation(
        tenant_id=tenant_id,
        invitation_code=invitation_code,
        created_by_id=user.id,
        notes=notes,
        max_uses=int(request.form.get('max_uses', 1)),
        expires_at=None  # Could add expiry date in future
    )
    
    db.session.add(invitation)
    db.session.commit()
    
    flash(f'Invitation code created: {invitation_code}', 'success')
    return redirect(url_for('tenant_access_management'))


@app.route('/admin/tenant-access/revoke/<int:access_id>', methods=['POST'])
def revoke_tenant_access(access_id):
    """Revoke organization user's access"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccess
    
    tenant_id = get_current_tenant_id()
    access = TenantAccess.query.get_or_404(access_id)
    
    if access.tenant_id != tenant_id:
        flash('You can only revoke access for your own tenant', 'error')
        return redirect(url_for('tenant_access_management'))
    
    access.is_active = False
    db.session.commit()
    
    # Send notification message to the organization user
    org_user = User.query.get(access.organization_user_id)
    if org_user:
        from models import DirectMessage, DirectMessageRecipient
        message = DirectMessage(
            subject=f'Access to {Tenant.query.get(tenant_id).name} revoked',
            content=f'Your access to {Tenant.query.get(tenant_id).name} has been revoked by {user.username}.',
            sender_id=user.id,
            tenant_id=tenant_id,
            message_type='general'
        )
        db.session.add(message)
        db.session.flush()
        
        recipient = DirectMessageRecipient(message_id=message.id, recipient_id=org_user.id)
        db.session.add(recipient)
        db.session.commit()
    
    flash('Access revoked successfully', 'success')
    return redirect(url_for('tenant_access_management'))


@app.route('/admin/tenant-access/approve/<int:request_id>', methods=['POST'])
def approve_tenant_access_request(request_id):
    """Approve organization user's request"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccess, TenantAccessRequest, DirectMessage, DirectMessageRecipient
    
    tenant_id = get_current_tenant_id()
    request_obj = TenantAccessRequest.query.get_or_404(request_id)
    
    if request_obj.tenant_id != tenant_id:
        flash('You can only approve requests for your own tenant', 'error')
        return redirect(url_for('tenant_access_management'))
    
    if request_obj.status != 'pending':
        flash('This request has already been processed', 'error')
        return redirect(url_for('tenant_access_management'))
    
    # Check if access already exists
    existing_access = TenantAccess.query.filter_by(
        tenant_id=tenant_id,
        organization_user_id=request_obj.requested_by_id,
        is_active=True
    ).first()
    
    if existing_access:
        # Just update the request status
        request_obj.status = 'approved'
        request_obj.reviewed_by_id = user.id
        request_obj.reviewed_at = datetime.utcnow()
        db.session.commit()
        flash('Access already exists for this user', 'info')
    else:
        # Create new access
        access = TenantAccess(
            tenant_id=tenant_id,
            organization_user_id=request_obj.requested_by_id,
            granted_by_id=user.id,
            notes=request_obj.notes
        )
        db.session.add(access)
        
        # Update request status
        request_obj.status = 'approved'
        request_obj.reviewed_by_id = user.id
        request_obj.reviewed_at = datetime.utcnow()
        db.session.commit()
        
        # Send notification message
        org_user = User.query.get(request_obj.requested_by_id)
        if org_user:
            tenant = Tenant.query.get(tenant_id)
            message = DirectMessage(
                subject=f'Access to {tenant.name} approved',
                content=f'Your request to access {tenant.name} has been approved by {user.username}.',
                sender_id=user.id,
                tenant_id=tenant_id,
                message_type='general'
            )
            db.session.add(message)
            db.session.flush()
            
            recipient = DirectMessageRecipient(message_id=message.id, recipient_id=org_user.id)
            db.session.add(recipient)
            db.session.commit()
        
        flash('Access request approved', 'success')
    
    return redirect(url_for('tenant_access_management'))


@app.route('/admin/tenant-access/deny/<int:request_id>', methods=['POST'])
def deny_tenant_access_request(request_id):
    """Deny organization user's request"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccessRequest, DirectMessage, DirectMessageRecipient
    
    tenant_id = get_current_tenant_id()
    request_obj = TenantAccessRequest.query.get_or_404(request_id)
    
    if request_obj.tenant_id != tenant_id:
        flash('You can only deny requests for your own tenant', 'error')
        return redirect(url_for('tenant_access_management'))
    
    if request_obj.status != 'pending':
        flash('This request has already been processed', 'error')
        return redirect(url_for('tenant_access_management'))
    
    denial_reason = request.form.get('denial_reason', '').strip()
    
    # Update request status
    request_obj.status = 'denied'
    request_obj.reviewed_by_id = user.id
    request_obj.reviewed_at = datetime.utcnow()
    if denial_reason:
        request_obj.notes = f"Denial reason: {denial_reason}"
    db.session.commit()
    
    # Send notification message
    org_user = User.query.get(request_obj.requested_by_id)
    if org_user:
        tenant = Tenant.query.get(tenant_id)
        message = DirectMessage(
            subject=f'Access to {tenant.name} denied',
            content=f'Your request to access {tenant.name} has been denied by {user.username}.{(" Reason: " + denial_reason) if denial_reason else ""}',
            sender_id=user.id,
            tenant_id=tenant_id,
            message_type='general'
        )
        db.session.add(message)
        db.session.flush()
        
        recipient = DirectMessageRecipient(message_id=message.id, recipient_id=org_user.id)
        db.session.add(recipient)
        db.session.commit()
    
    flash('Access request denied', 'success')
    return redirect(url_for('tenant_access_management'))


@app.route('/tenant/request-access', methods=['GET', 'POST'])
def request_tenant_access():
    """Organization user: Request access to a tenant by ID"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or user.role != 'organization':
        flash('This feature is only available for organization users', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccessRequest, TenantAccess
    
    if request.method == 'POST':
        tenant_id_str = request.form.get('tenant_id', '').strip()
        notes = request.form.get('notes', '').strip()
        
        try:
            tenant_id = int(tenant_id_str)
        except ValueError:
            flash('Invalid tenant ID', 'error')
            return redirect(url_for('request_tenant_access'))
        
        # Check if tenant exists
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            flash('Tenant not found', 'error')
            return redirect(url_for('request_tenant_access'))
        
        # Check if access already exists
        existing_access = TenantAccess.query.filter_by(
            tenant_id=tenant_id,
            organization_user_id=user.id,
            is_active=True
        ).first()
        
        if existing_access:
            flash('You already have access to this tenant', 'info')
            return redirect(url_for('my_tenant_access'))
        
        # Check if there's already a pending request
        existing_request = TenantAccessRequest.query.filter_by(
            tenant_id=tenant_id,
            requested_by_id=user.id,
            status='pending'
        ).first()
        
        if existing_request:
            flash('You already have a pending request for this tenant', 'info')
            return redirect(url_for('request_tenant_access'))
        
        # Create request
        access_request = TenantAccessRequest(
            tenant_id=tenant_id,
            requested_by_id=user.id,
            notes=notes
        )
        db.session.add(access_request)
        db.session.commit()
        
        # Notify tenant admins
        tenant_admins = User.query.filter_by(
            role='admin',
            is_admin=True
        ).all()
        
        from models import DirectMessage, DirectMessageRecipient
        for admin in tenant_admins:
            admin_tenant_id = get_tenant_id_for_user(admin.id)
            if admin_tenant_id == tenant_id:
                message = DirectMessage(
                    subject=f'New access request for {tenant.name}',
                    content=f'{user.username} ({user.full_name or user.email}) has requested access to {tenant.name}.\n\nNotes: {notes if notes else "No notes provided"}',
                    sender_id=user.id,
                    tenant_id=tenant_id,
                    message_type='general',
                    is_urgent=False
                )
                db.session.add(message)
                db.session.flush()
                
                recipient = DirectMessageRecipient(message_id=message.id, recipient_id=admin.id)
                db.session.add(recipient)
        
        db.session.commit()
        
        flash('Access request submitted. The tenant admin will be notified.', 'success')
        return redirect(url_for('my_tenant_access'))
    
    return render_template('tenant/request_access.html')


@app.route('/tenant/join/<invitation_code>')
def join_tenant_via_invitation(invitation_code):
    """Organization user: Accept invitation link"""
    if 'carer_id' not in session:
        flash('Please log in to accept this invitation', 'info')
        session['invitation_code'] = invitation_code  # Store for after login
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or user.role != 'organization':
        flash('This feature is only available for organization users', 'error')
        return redirect(url_for('index'))
    
    from models import TenantInvitation, TenantAccess
    
    invitation = TenantInvitation.query.filter_by(invitation_code=invitation_code).first()
    
    if not invitation:
        flash('Invalid invitation code', 'error')
        return redirect(url_for('my_tenant_access'))
    
    if not invitation.is_valid():
        flash('This invitation has expired or is no longer valid', 'error')
        return redirect(url_for('my_tenant_access'))
    
    # Check if access already exists
    existing_access = TenantAccess.query.filter_by(
        tenant_id=invitation.tenant_id,
        organization_user_id=user.id,
        is_active=True
    ).first()
    
    if existing_access:
        flash('You already have access to this tenant', 'info')
        return redirect(url_for('my_tenant_access'))
    
    # Create access
    access = TenantAccess(
        tenant_id=invitation.tenant_id,
        organization_user_id=user.id,
        granted_by_id=invitation.created_by_id,
        notes=f'Joined via invitation: {invitation_code}'
    )
    db.session.add(access)
    
    # Update invitation usage
    invitation.used_count += 1
    if invitation.used_count >= invitation.max_uses:
        invitation.is_active = False
    
    db.session.commit()
    
    tenant = Tenant.query.get(invitation.tenant_id)
    flash(f'Successfully joined {tenant.name}!', 'success')
    return redirect(url_for('my_tenant_access'))


@app.route('/tenant/switch/<int:tenant_id>')
def switch_tenant(tenant_id):
    """Switch active tenant (for org users with multiple access)"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or user.role != 'organization':
        flash('This feature is only available for organization users', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccess
    
    # Verify user has access to this tenant
    access = TenantAccess.query.filter_by(
        tenant_id=tenant_id,
        organization_user_id=user.id,
        is_active=True
    ).first()
    
    if not access:
        flash('You do not have access to this tenant', 'error')
        return redirect(url_for('my_tenant_access'))
    
    # Store in session
    session['active_tenant_id'] = tenant_id
    tenant = Tenant.query.get(tenant_id)
    flash(f'Switched to {tenant.name}', 'success')
    
    return redirect(request.referrer or url_for('index'))


@app.route('/tenant/my-access')
def my_tenant_access():
    """Organization user: View all tenants they can access"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or user.role != 'organization':
        flash('This feature is only available for organization users', 'error')
        return redirect(url_for('index'))
    
    from models import TenantAccess
    
    accesses = TenantAccess.query.filter_by(
        organization_user_id=user.id,
        is_active=True
    ).order_by(TenantAccess.granted_at.desc()).all()
    
    current_tenant_id = session.get('active_tenant_id', get_current_tenant_id())
    
    return render_template('tenant/my_access.html',
                         accesses=accesses,
                         current_tenant_id=current_tenant_id)


# ========== BOOKING/SCHEDULING SYSTEM ROUTES ==========

@app.route('/shifts')
def shifts_calendar():
    """Calendar view of shifts (weekly grid for desktop, agenda for mobile)"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    # Check feature flag - booking.view is available for Pro+ or optional for Family
    if not check_feature_enabled(tenant_id, 'booking.view'):
        flash('Shift booking is only available for Professional tier subscriptions.', 'info')
        return redirect(url_for('index'))
    
    # Get date range (default to current week)
    from datetime import timedelta
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)  # Sunday
    
    start_date = request.args.get('start_date', week_start.isoformat())
    end_date = request.args.get('end_date', week_end.isoformat())
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        start_dt = datetime.combine(week_start, datetime.min.time())
        end_dt = datetime.combine(week_end, datetime.max.time())
    
    # Get shifts in date range
    shifts = Shift.query.filter(
        Shift.tenant_id == tenant_id,
        Shift.start_datetime >= start_dt,
        Shift.start_datetime <= end_dt + timedelta(days=1)
    ).order_by(Shift.start_datetime.asc()).all()
    
    # Get user's shifts if carer
    my_shifts = []
    if user and user.role == 'carer':
        my_shifts = Shift.query.filter(
            Shift.tenant_id == tenant_id,
            Shift.assigned_carer_id == user.id,
            Shift.start_datetime >= start_dt,
            Shift.start_datetime <= end_dt + timedelta(days=1)
        ).order_by(Shift.start_datetime.asc()).all()
    
    # Group shifts by date for calendar view
    shifts_by_date = {}
    for shift in shifts:
        shift_date = shift.start_datetime.date()
        if shift_date not in shifts_by_date:
            shifts_by_date[shift_date] = []
        shifts_by_date[shift_date].append(shift)
    
    # Generate list of dates for the week
    week_dates = []
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        week_dates.append(current_date)
        current_date = current_date + timedelta(days=1)
    
    return render_template('shifts/calendar.html',
                         shifts=shifts,
                         shifts_by_date=shifts_by_date,
                         week_dates=week_dates,
                         my_shifts=my_shifts,
                         start_date=start_dt.date(),
                         end_date=end_dt.date(),
                         user=user,
                         can_manage=check_feature_enabled(tenant_id, 'booking.manage'))


@app.route('/shifts/new', methods=['GET', 'POST'])
def new_shift():
    """Create a new shift"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    # Check feature flag - booking.manage is Pro+ only
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Shift management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title', '').strip()
            service_id = request.form.get('service_id', '').strip()
            schedule_id = request.form.get('schedule_id', '').strip()
            assigned_carer_id = request.form.get('assigned_carer_id', '').strip()
            
            # Parse datetime
            start_datetime_str = request.form.get('start_datetime', '').strip()
            end_datetime_str = request.form.get('end_datetime', '').strip()
            
            if not title or not start_datetime_str or not end_datetime_str:
                flash('Title, start time, and end time are required', 'error')
                return redirect(url_for('new_shift'))
            
            start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M')
            end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%dT%H:%M')
            
            if end_datetime <= start_datetime:
                flash('End time must be after start time', 'error')
                return redirect(url_for('new_shift'))
            
            # Calculate duration
            duration = (end_datetime - start_datetime).total_seconds() / 3600  # hours
            
            # Create shift
            shift = Shift(
                tenant_id=tenant_id,
                title=title,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                duration_hours=duration,
                service_id=int(service_id) if service_id and service_id.isdigit() else None,
                schedule_id=int(schedule_id) if schedule_id and schedule_id.isdigit() else None,
                assigned_carer_id=int(assigned_carer_id) if assigned_carer_id and assigned_carer_id.isdigit() else None,
                location=request.form.get('location', '').strip() or None,
                notes=request.form.get('notes', '').strip() or None,
                appointment_type=request.form.get('appointment_type', 'service'),
                created_by=user.id,
                booking_source='web'
            )
            
            db.session.add(shift)
            db.session.flush()  # Get shift.id
            
            # Create history entry
            history = ShiftHistory(
                shift_id=shift.id,
                action='created',
                changed_by=user.id,
                changes=f'{{"title": "{title}", "start": "{start_datetime_str}", "end": "{end_datetime_str}"}}'
            )
            db.session.add(history)
            db.session.commit()
            
            flash(f'Shift "{title}" created successfully!', 'success')
            return redirect(url_for('view_shift', shift_id=shift.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating shift: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
    
    # GET request - show form
    services = Service.query.filter_by(tenant_id=tenant_id, is_active=True).order_by(Service.display_order, Service.name).all()
    schedules = Schedule.query.filter_by(tenant_id=tenant_id, is_active=True).order_by(Schedule.name).all()
    carers = User.query.filter_by(role='carer', is_approved=True).all()
    
    # Default to today, current hour
    from datetime import timedelta
    now = datetime.now()
    default_start = now.replace(minute=0, second=0, microsecond=0)
    default_end = default_start + timedelta(hours=1)
    
    return render_template('shifts/new_shift.html',
                         services=services,
                         schedules=schedules,
                         carers=carers,
                         default_start=default_start.strftime('%Y-%m-%dT%H:%M'),
                         default_end=default_end.strftime('%Y-%m-%dT%H:%M'))


@app.route('/shifts/<int:shift_id>')
def view_shift(shift_id):
    """View shift details"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    shift = Shift.query.get_or_404(shift_id)
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to view this shift.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.view'):
        flash('Shift booking is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('index'))
    
    can_manage = check_feature_enabled(tenant_id, 'booking.manage')
    can_swap = check_feature_enabled(tenant_id, 'booking.swap')
    
    return render_template('shifts/view_shift.html',
                         shift=shift,
                         user=user,
                         can_manage=can_manage,
                         can_swap=can_swap)


@app.route('/shifts/<int:shift_id>/edit', methods=['GET', 'POST'])
def edit_shift(shift_id):
    """Edit a shift"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    shift = Shift.query.get_or_404(shift_id)
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to edit this shift.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Shift management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    # Check permissions (admin or assigned carer)
    if not current_user_is_admin() and shift.assigned_carer_id != user.id:
        flash('You can only edit shifts assigned to you, or you must be an admin.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    if request.method == 'POST':
        try:
            # Store old values for history
            old_title = shift.title
            old_start = shift.start_datetime
            old_end = shift.end_datetime
            old_carer = shift.assigned_carer_id
            
            # Update shift
            shift.title = request.form.get('title', '').strip()
            start_datetime_str = request.form.get('start_datetime', '').strip()
            end_datetime_str = request.form.get('end_datetime', '').strip()
            
            if start_datetime_str and end_datetime_str:
                shift.start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M')
                shift.end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%dT%H:%M')
                shift.duration_hours = (shift.end_datetime - shift.start_datetime).total_seconds() / 3600
            
            service_id = request.form.get('service_id', '').strip()
            shift.service_id = int(service_id) if service_id and service_id.isdigit() else None
            
            schedule_id = request.form.get('schedule_id', '').strip()
            shift.schedule_id = int(schedule_id) if schedule_id and schedule_id.isdigit() else None
            
            assigned_carer_id = request.form.get('assigned_carer_id', '').strip()
            shift.assigned_carer_id = int(assigned_carer_id) if assigned_carer_id and assigned_carer_id.isdigit() else None
            
            shift.location = request.form.get('location', '').strip() or None
            shift.notes = request.form.get('notes', '').strip() or None
            shift.appointment_type = request.form.get('appointment_type', 'service')
            
            # Create history entry
            changes = []
            if old_title != shift.title:
                changes.append(f'title: "{old_title}" → "{shift.title}"')
            if old_start != shift.start_datetime:
                changes.append(f'start: "{old_start}" → "{shift.start_datetime}"')
            if old_end != shift.end_datetime:
                changes.append(f'end: "{old_end}" → "{shift.end_datetime}"')
            if old_carer != shift.assigned_carer_id:
                changes.append(f'assigned_carer: {old_carer} → {shift.assigned_carer_id}')
            
            if changes:
                history = ShiftHistory(
                    shift_id=shift.id,
                    action='updated',
                    changed_by=user.id,
                    changes='; '.join(changes)
                )
                db.session.add(history)
            
            db.session.commit()
            flash('Shift updated successfully!', 'success')
            return redirect(url_for('view_shift', shift_id=shift_id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating shift: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
    
    # GET request - show edit form
    services = Service.query.filter_by(tenant_id=tenant_id, is_active=True).order_by(Service.display_order, Service.name).all()
    schedules = Schedule.query.filter_by(tenant_id=tenant_id, is_active=True).order_by(Schedule.name).all()
    carers = User.query.filter_by(role='carer', is_approved=True).all()
    
    return render_template('shifts/edit_shift.html',
                         shift=shift,
                         services=services,
                         schedules=schedules,
                         carers=carers)


@app.route('/shifts/<int:shift_id>/delete', methods=['POST'])
def delete_shift(shift_id):
    """Delete a shift"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    shift = Shift.query.get_or_404(shift_id)
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to delete this shift.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Shift management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    # Only admins can delete
    if not current_user_is_admin():
        flash('Only admins can delete shifts.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    try:
        title = shift.title
        
        # Create history entry before deletion
        history = ShiftHistory(
            shift_id=shift.id,
            action='deleted',
            changed_by=user.id,
            changes=f'Shift "{title}" deleted'
        )
        db.session.add(history)
        
        # Delete shift (cascade will handle related records)
        db.session.delete(shift)
        db.session.commit()
        
        flash(f'Shift "{title}" deleted successfully', 'success')
        return redirect(url_for('shifts_calendar'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting shift: {str(e)}', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))


@app.route('/shifts/my-shifts')
def my_shifts():
    """Carer's personal shift view"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.view'):
        flash('Shift booking is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('index'))
    
    # Get user's shifts (upcoming and past)
    upcoming_shifts = Shift.query.filter(
        Shift.tenant_id == tenant_id,
        Shift.assigned_carer_id == user.id,
        Shift.start_datetime >= datetime.now(),
        Shift.status != 'cancelled'
    ).order_by(Shift.start_datetime.asc()).all()
    
    past_shifts = Shift.query.filter(
        Shift.tenant_id == tenant_id,
        Shift.assigned_carer_id == user.id,
        Shift.start_datetime < datetime.now()
    ).order_by(Shift.start_datetime.desc()).limit(20).all()
    
    return render_template('shifts/my_shifts.html',
                         upcoming_shifts=upcoming_shifts,
                         past_shifts=past_shifts,
                         user=user)


@app.route('/shifts/<int:shift_id>/complete', methods=['POST'])
def complete_shift(shift_id):
    """Mark shift as completed (carer)"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    shift = Shift.query.get_or_404(shift_id)
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to complete this shift.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Shift management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    # Only assigned carer can complete
    if shift.assigned_carer_id != user.id:
        flash('You can only complete shifts assigned to you.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    try:
        shift.is_completed = True
        shift.completed_at = datetime.utcnow()
        shift.completed_by = user.id
        shift.status = 'completed'
        
        # Create history entry
        history = ShiftHistory(
            shift_id=shift.id,
            action='completed',
            changed_by=user.id,
            changes=f'Shift marked as completed at {shift.completed_at}'
        )
        db.session.add(history)
        db.session.commit()
        
        flash('Shift marked as completed!', 'success')
        return redirect(url_for('view_shift', shift_id=shift_id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error completing shift: {str(e)}', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))


@app.route('/admin/services', methods=['GET'])
def admin_services():
    """Admin: List all services"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Service management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    services = Service.query.filter_by(tenant_id=tenant_id).order_by(Service.display_order, Service.name).all()
    
    return render_template('admin/services.html', services=services)


@app.route('/admin/services/new', methods=['GET', 'POST'])
def admin_service_new():
    """Admin: Create new service"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Service management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    if request.method == 'POST':
        try:
            service = Service(
                tenant_id=tenant_id,
                name=request.form.get('name', '').strip(),
                duration_minutes=int(request.form.get('duration_minutes', 30)),
                category=request.form.get('category', '').strip() or None,
                description=request.form.get('description', '').strip() or None,
                price=float(request.form.get('price', 0) or 0),
                display_order=int(request.form.get('display_order', 0) or 0),
                is_active=True
            )
            db.session.add(service)
            db.session.commit()
            flash(f'Service "{service.name}" created successfully!', 'success')
            return redirect(url_for('admin_services'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating service: {str(e)}', 'error')
    
    return render_template('admin/service_edit.html', service=None)


@app.route('/admin/services/<int:service_id>/edit', methods=['GET', 'POST'])
def admin_service_edit(service_id):
    """Admin: Edit service"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    service = Service.query.get_or_404(service_id)
    
    # Check tenant access
    if service.tenant_id != tenant_id:
        flash('You do not have permission to edit this service.', 'error')
        return redirect(url_for('admin_services'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Service management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    if request.method == 'POST':
        try:
            service.name = request.form.get('name', '').strip()
            service.duration_minutes = int(request.form.get('duration_minutes', 30))
            service.category = request.form.get('category', '').strip() or None
            service.description = request.form.get('description', '').strip() or None
            service.price = float(request.form.get('price', 0) or 0)
            service.display_order = int(request.form.get('display_order', 0) or 0)
            service.is_active = request.form.get('is_active') == 'on'
            db.session.commit()
            flash(f'Service "{service.name}" updated successfully!', 'success')
            return redirect(url_for('admin_services'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating service: {str(e)}', 'error')
    
    return render_template('admin/service_edit.html', service=service)


@app.route('/admin/schedules', methods=['GET'])
def admin_schedules():
    """Admin: List all schedules"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Schedule management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    schedules = Schedule.query.filter_by(tenant_id=tenant_id).order_by(Schedule.name).all()
    
    return render_template('admin/schedules.html', schedules=schedules)


@app.route('/admin/schedules/new', methods=['GET', 'POST'])
def admin_schedule_new():
    """Admin: Create new schedule"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Schedule management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    if request.method == 'POST':
        try:
            schedule = Schedule(
                tenant_id=tenant_id,
                name=request.form.get('name', '').strip(),
                short_code=request.form.get('short_code', '').strip() or None,
                schedule_type=request.form.get('schedule_type', '').strip() or None,
                location=request.form.get('location', '').strip() or None,
                color_code=request.form.get('color_code', '').strip() or None,
                is_active=True
            )
            db.session.add(schedule)
            db.session.commit()
            flash(f'Schedule "{schedule.name}" created successfully!', 'success')
            return redirect(url_for('admin_schedules'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating schedule: {str(e)}', 'error')
    
    return render_template('admin/schedule_edit.html', schedule=None)


@app.route('/admin/schedules/<int:schedule_id>/edit', methods=['GET', 'POST'])
def admin_schedule_edit(schedule_id):
    """Admin: Edit schedule"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    if not user or not current_user_is_admin():
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    tenant_id = get_current_tenant_id()
    schedule = Schedule.query.get_or_404(schedule_id)
    
    # Check tenant access
    if schedule.tenant_id != tenant_id:
        flash('You do not have permission to edit this schedule.', 'error')
        return redirect(url_for('admin_schedules'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.manage'):
        flash('Schedule management is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('admin'))
    
    if request.method == 'POST':
        try:
            schedule.name = request.form.get('name', '').strip()
            schedule.short_code = request.form.get('short_code', '').strip() or None
            schedule.schedule_type = request.form.get('schedule_type', '').strip() or None
            schedule.location = request.form.get('location', '').strip() or None
            schedule.color_code = request.form.get('color_code', '').strip() or None
            schedule.is_active = request.form.get('is_active') == 'on'
            db.session.commit()
            flash(f'Schedule "{schedule.name}" updated successfully!', 'success')
            return redirect(url_for('admin_schedules'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating schedule: {str(e)}', 'error')
    
    return render_template('admin/schedule_edit.html', schedule=schedule)


@app.route('/shifts/<int:shift_id>/request-swap', methods=['GET', 'POST'])
def request_shift_swap(shift_id):
    """Request to swap a shift"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    shift = Shift.query.get_or_404(shift_id)
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to swap this shift.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.swap'):
        flash('Shift swapping is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    # Only assigned carer can request swap
    if shift.assigned_carer_id != user.id:
        flash('You can only request swaps for shifts assigned to you.', 'error')
        return redirect(url_for('view_shift', shift_id=shift_id))
    
    if request.method == 'POST':
        try:
            target_carer_id = request.form.get('target_carer_id', '').strip()
            message = request.form.get('message', '').strip()
            
            swap_request = ShiftSwapRequest(
                shift_id=shift.id,
                requester_id=user.id,
                target_carer_id=int(target_carer_id) if target_carer_id and target_carer_id.isdigit() else None,
                message=message or None,
                status='pending'
            )
            db.session.add(swap_request)
            db.session.commit()
            
            # Send notification message to target carer (if specified) or all carers
            from models import DirectMessage, DirectMessageRecipient
            if swap_request.target_carer_id:
                # Send to specific carer
                target_carer = User.query.get(swap_request.target_carer_id)
                if target_carer:
                    dm = DirectMessage(
                        subject=f'Shift Swap Request: {shift.title}',
                        content=f'{user.username} has requested to swap the following shift:\n\n{shift.title}\n{shift.start_datetime.strftime("%d %b %Y at %I:%M %p")}\n\n{message if message else "No message provided"}',
                        sender_id=user.id,
                        recipient_id=target_carer.id,  # Set recipient_id for backward compatibility
                        tenant_id=tenant_id,
                        message_type='shift_request',
                        related_shift_id=shift.id,
                        is_urgent=False
                    )
                    db.session.add(dm)
                    db.session.flush()
                    recipient = DirectMessageRecipient(message_id=dm.id, recipient_id=target_carer.id)
                    db.session.add(recipient)
            else:
                # Send to all carers (open request) - filter by tenant
                from models import TenantAccess
                tenant_user_ids = [ta.user_id for ta in TenantAccess.query.filter_by(tenant_id=tenant_id).all()]
                all_carers = User.query.filter(
                    User.id.in_(tenant_user_ids),
                    User.role == 'carer',
                    User.is_approved == True
                ).all() if tenant_user_ids else []
                for carer in all_carers:
                    if carer.id != user.id:
                        dm = DirectMessage(
                            subject=f'Open Shift Swap Request: {shift.title}',
                            content=f'{user.username} has requested to swap the following shift:\n\n{shift.title}\n{shift.start_datetime.strftime("%d %b %Y at %I:%M %p")}\n\n{message if message else "No message provided"}',
                            sender_id=user.id,
                            recipient_id=carer.id,  # Set recipient_id for backward compatibility
                            tenant_id=tenant_id,
                            message_type='shift_request',
                            related_shift_id=shift.id,
                            is_urgent=False
                        )
                        db.session.add(dm)
                        db.session.flush()
                        recipient = DirectMessageRecipient(message_id=dm.id, recipient_id=carer.id)
                        db.session.add(recipient)
            
            db.session.commit()
            flash('Shift swap request sent!', 'success')
            return redirect(url_for('view_shift', shift_id=shift_id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating swap request: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
    
    # GET request - show form
    all_carers = User.query.filter_by(role='carer', is_approved=True).all()
    # Exclude the current user
    available_carers = [c for c in all_carers if c.id != user.id]
    
    return render_template('shifts/request_swap.html',
                         shift=shift,
                         available_carers=available_carers)


@app.route('/shifts/swap-requests')
def shift_swap_requests():
    """View all swap requests"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.swap'):
        flash('Shift swapping is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('shifts_calendar'))
    
    is_admin = current_user_is_admin()
    
    if is_admin:
        # Admins see all pending swap requests in their tenant
        all_requests = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
            Shift.tenant_id == tenant_id,
            ShiftSwapRequest.status == 'pending'
        ).order_by(ShiftSwapRequest.created_at.desc()).all()
        
        return render_template('shifts/swap_requests.html',
                             my_requests=[],
                             requests_to_me=[],
                             open_requests=all_requests,
                             all_requests=all_requests,  # Pass all requests for admin view
                             user=user,
                             is_admin=True)
    else:
        # Regular users see only requests directed to them or open requests
        my_requests = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
            Shift.tenant_id == tenant_id,
            ShiftSwapRequest.requester_id == user.id,
            ShiftSwapRequest.status == 'pending'
        ).order_by(ShiftSwapRequest.created_at.desc()).all()
        
        requests_to_me = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
            Shift.tenant_id == tenant_id,
            ShiftSwapRequest.target_carer_id == user.id,
            ShiftSwapRequest.status == 'pending'
        ).order_by(ShiftSwapRequest.created_at.desc()).all()
        
        # Open requests (no target specified)
        open_requests = ShiftSwapRequest.query.join(Shift, ShiftSwapRequest.shift_id == Shift.id).filter(
            Shift.tenant_id == tenant_id,
            ShiftSwapRequest.target_carer_id == None,
            ShiftSwapRequest.status == 'pending',
            ShiftSwapRequest.requester_id != user.id
        ).order_by(ShiftSwapRequest.created_at.desc()).all()
        
        return render_template('shifts/swap_requests.html',
                             my_requests=my_requests,
                             requests_to_me=requests_to_me,
                             open_requests=open_requests,
                             user=user,
                             is_admin=False)


@app.route('/shifts/swap-requests/<int:request_id>/approve', methods=['POST'])
def approve_shift_swap(request_id):
    """Approve a shift swap request"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    swap_request = ShiftSwapRequest.query.get_or_404(request_id)
    shift = swap_request.shift
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to approve this swap.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.swap'):
        flash('Shift swapping is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    # Target carer, admin, or anyone for open requests can approve
    if swap_request.target_carer_id is None:
        # Open request - anyone can accept
        pass
    elif swap_request.target_carer_id != user.id and not current_user_is_admin():
        # Specific request - only target carer or admin can approve
        flash('You can only approve swap requests directed to you, or you must be an admin.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    if swap_request.status != 'pending':
        flash('This swap request has already been processed.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    try:
        # Swap the assigned carer
        old_carer_id = shift.assigned_carer_id
        shift.assigned_carer_id = swap_request.requester_id
        
        # Update swap request
        swap_request.status = 'approved'
        swap_request.responded_at = datetime.utcnow()
        swap_request.responded_by = user.id
        
        # Create history entry
        history = ShiftHistory(
            shift_id=shift.id,
            action='assigned',
            changed_by=user.id,
            changes=f'Shift swapped: carer {old_carer_id} → {swap_request.requester_id} (approved by {user.username})'
        )
        db.session.add(history)
        
        # Send notification to requester
        from models import DirectMessage, DirectMessageRecipient
        requester = User.query.get(swap_request.requester_id)
        if requester:
            dm = DirectMessage(
                subject=f'Shift Swap Approved: {shift.title}',
                content=f'Your swap request for "{shift.title}" has been approved by {user.username}.',
                sender_id=user.id,
                tenant_id=tenant_id,
                message_type='shift_request',
                is_urgent=False
            )
            db.session.add(dm)
            db.session.flush()
            recipient = DirectMessageRecipient(message_id=dm.id, recipient_id=requester.id)
            db.session.add(recipient)
        
        db.session.commit()
        flash('Shift swap approved!', 'success')
        return redirect(url_for('shift_swap_requests'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error approving swap: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('shift_swap_requests'))


@app.route('/shifts/swap-requests/<int:request_id>/reject', methods=['POST'])
def reject_shift_swap(request_id):
    """Reject a shift swap request"""
    if 'carer_id' not in session:
        return redirect(url_for('login'))
    
    user = get_current_user()
    tenant_id = get_current_tenant_id()
    
    swap_request = ShiftSwapRequest.query.get_or_404(request_id)
    shift = swap_request.shift
    
    # Check tenant access
    if shift.tenant_id != tenant_id:
        flash('You do not have permission to reject this swap.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    # Check feature flag
    if not check_feature_enabled(tenant_id, 'booking.swap'):
        flash('Shift swapping is only available for Professional tier subscriptions.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    # Only target carer or admin can reject
    if swap_request.target_carer_id != user.id and not current_user_is_admin():
        flash('You can only reject swap requests directed to you, or you must be an admin.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    if swap_request.status != 'pending':
        flash('This swap request has already been processed.', 'error')
        return redirect(url_for('shift_swap_requests'))
    
    try:
        # Update swap request
        swap_request.status = 'rejected'
        swap_request.responded_at = datetime.utcnow()
        swap_request.responded_by = user.id
        
        # Send notification to requester
        from models import DirectMessage, DirectMessageRecipient
        requester = User.query.get(swap_request.requester_id)
        if requester:
            dm = DirectMessage(
                subject=f'Shift Swap Rejected: {shift.title}',
                content=f'Your swap request for "{shift.title}" has been rejected by {user.username}.',
                sender_id=user.id,
                tenant_id=tenant_id,
                message_type='shift_request',
                is_urgent=False
            )
            db.session.add(dm)
            db.session.flush()
            recipient = DirectMessageRecipient(message_id=dm.id, recipient_id=requester.id)
            db.session.add(recipient)
        
        db.session.commit()
        flash('Shift swap rejected.', 'info')
        return redirect(url_for('shift_swap_requests'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error rejecting swap: {str(e)}', 'error')
        return redirect(url_for('shift_swap_requests'))


# ========== END OF NEW ROUTES ==========

if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'static/uploads_test'), exist_ok=True)
    with app.app_context():
        db.create_all()
        add_missing_columns()
        add_user_profile_columns()
        seed_gas_goals()  # NEW: seed Tamara's GAS goals once

    app.run(host='0.0.0.0', port=5001, debug=True)