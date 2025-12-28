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
    focus_task_id = db.Column(db.Integer, db.ForeignKey('focus_tasks.id'), nullable=True)
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
    is_admin = db.Column(db.Boolean, default=False)  # Legacy flag - kept for backward compatibility
    is_approved = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(50), default='carer')  
    # Values: 'super_user' | 'admin' | 'professional' | 'organization' | 'carer' | 'readonly'
    # super_user: Dev team / highest level (all permissions)
    # admin: Family/parent admins (manage carers, activities, view all logs)
    # professional: OTs, Speech, Behavioral therapists (view logs, create focused tasks, analytics)
    # organization: Day care programs, external orgs (view own logs, limited analytics)
    # carer: Support workers (create logs, view own logs, message others)
    # readonly: Read-only access (view logs, view analytics, no editing)
    
    # Additional role-specific fields
    is_readonly = db.Column(db.Boolean, default=False)  # Quick flag for read-only access
    professional_type = db.Column(db.String(100), nullable=True)  
    # For professionals: 'OT', 'Speech', 'Behavioral', 'Physio', 'Psychologist', etc.
    organization_id = db.Column(db.Integer, nullable=True)  
    # For organization users - links them to their organization (future: FK to organizations table)
    
    # Profile fields
    full_name = db.Column(db.String(200), nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    education = db.Column(db.Text, nullable=True)  # Education background
    qualifications = db.Column(db.Text, nullable=True)  # Certifications, licenses
    skills = db.Column(db.Text, nullable=True)  # Skills and competencies
    strengths = db.Column(db.Text, nullable=True)  # What they're good at
    experience_years = db.Column(db.Integer, nullable=True)  # Years of experience
    specializations = db.Column(db.Text, nullable=True)  # Special areas of expertise
    bio = db.Column(db.Text, nullable=True)  # General bio/notes
    hire_date = db.Column(db.Date, nullable=True)  # When they started
    notes = db.Column(db.Text, nullable=True)  # Additional admin notes
    
    # Password reset fields
    reset_token = db.Column(db.String(100), nullable=True)  # Token for password reset
    reset_token_expiry = db.Column(db.DateTime, nullable=True)  # When token expires

class InsuranceRecord(db.Model):
    __tablename__ = 'insurance_records'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)

    public_liability_policy = db.Column(db.String(120))
    public_liability_insurer = db.Column(db.String(120))
    public_liability_coverage = db.Column(db.String(50))
    public_liability_expiry = db.Column(db.Date)

    professional_indemnity_policy = db.Column(db.String(120))
    professional_indemnity_insurer = db.Column(db.String(120))
    professional_indemnity_coverage = db.Column(db.String(50))
    professional_indemnity_expiry = db.Column(db.Date)

    workers_comp_policy = db.Column(db.String(120))
    workers_comp_insurer = db.Column(db.String(120))
    workers_comp_expiry = db.Column(db.Date)

    car_insurance_policy = db.Column(db.String(120))
    car_insurance_insurer = db.Column(db.String(120))
    car_insurance_expiry = db.Column(db.Date)

    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

class Notice(db.Model):
    __tablename__ = 'notices'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.String(100), nullable=False)  # username
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_important = db.Column(db.Boolean, default=False)  # For important notices that should be highlighted
    replies = db.relationship('NoticeReply', backref='notice', lazy=True, cascade="all, delete-orphan", order_by='NoticeReply.created_at')
    read_by = db.relationship('NoticeRead', backref='notice', lazy=True, cascade="all, delete-orphan")

class NoticeReply(db.Model):
    __tablename__ = 'notice_replies'
    
    id = db.Column(db.Integer, primary_key=True)
    notice_id = db.Column(db.Integer, db.ForeignKey('notices.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.String(100), nullable=False)  # username
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class NoticeRead(db.Model):
    __tablename__ = 'notice_reads'
    
    id = db.Column(db.Integer, primary_key=True)
    notice_id = db.Column(db.Integer, db.ForeignKey('notices.id'), nullable=False)
    user_id = db.Column(db.String(100), nullable=False)  # username
    read_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ensure a user can only have one read record per notice
    __table_args__ = (db.UniqueConstraint('notice_id', 'user_id', name='_notice_user_read_uc'),)

class CalendarEvent(db.Model):
    __tablename__ = 'calendar_events'
    
    id = db.Column(db.Integer, primary_key=True)
    google_event_id = db.Column(db.String(255), unique=True, nullable=False)  # Google's event ID (UID from iCal)
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_datetime = db.Column(db.DateTime, nullable=False)
    end_datetime = db.Column(db.DateTime, nullable=True)
    location = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<CalendarEvent {self.title} - {self.start_datetime}>'


# --- Focused Tasks (5-stage lifecycle) ---

class FocusTask(db.Model):
    __tablename__ = 'focus_tasks'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    short_code = db.Column(db.String(50), unique=True, nullable=True)
    description = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), default='active')  # active | paused | done
    focus_start = db.Column(db.Date, nullable=True)
    focus_end = db.Column(db.Date, nullable=True)
    created_by = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    auto_capture_activities = db.Column(db.Text, nullable=True)  # JSON array of activity names to auto-capture


class FocusStage(db.Model):
    __tablename__ = 'focus_stages'

    code = db.Column(db.String(10), primary_key=True)  # Q,W,H,C,Z
    short_name = db.Column(db.String(20), nullable=False)  # WHY/WHAT/HOW/CHECK/AUTO
    name = db.Column(db.String(100), nullable=False)
    order = db.Column(db.Integer, nullable=False)


class FocusEntry(db.Model):
    __tablename__ = 'focus_entries'

    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('focus_tasks.id'), nullable=False)
    stage_code = db.Column(db.String(10), db.ForeignKey('focus_stages.code'), nullable=False)
    entry_date = db.Column(db.DateTime, default=datetime.utcnow)
    title = db.Column(db.String(200), nullable=False)
    detail = db.Column(db.Text, nullable=False)
    added_by = db.Column(db.String(100), nullable=False)

    task = db.relationship('FocusTask', backref=db.backref('entries', lazy=True, cascade="all, delete-orphan"))
    stage = db.relationship('FocusStage')


# --- Feature Flag System (App Store / Multi-Tenant Support) ---

class Plan(db.Model):
    """Subscription plans (Starter, Family, Pro, Provider)"""
    __tablename__ = 'plans'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False)
    # Values: 'starter', 'family', 'pro', 'provider'
    
    display_name = db.Column(db.String(100), nullable=False)
    # "Starter", "Family Care", "Professional", "Provider"
    
    app_store_product_id = db.Column(db.String(200), nullable=True)
    # Apple App Store product ID for subscription (e.g., "com.tamaracare.family_monthly")
    
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenants = db.relationship('Tenant', backref='plan', lazy=True)
    features = db.relationship('PlanFeature', backref='plan', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Plan {self.code} - {self.display_name}>'


class Tenant(db.Model):
    """Organizations/families using the app"""
    __tablename__ = 'tenants'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('plans.id'), nullable=False)
    
    # For single-tenant: create one tenant with id=1, plan_id=1 (starter)
    # For multi-tenant: each organization gets a tenant
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Tenant {self.name} - Plan {self.plan_id}>'


class PlanFeature(db.Model):
    """Feature flags mapped to plans (single source of truth)"""
    __tablename__ = 'plan_features'
    
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('plans.id'), nullable=False)
    feature_key = db.Column(db.String(100), nullable=False)
    # Values from mapping table: 'care_log.create', 'messaging.family', 
    # 'booking.manage', 'activities.custom_limit', etc.
    
    feature_value = db.Column(db.String(200), nullable=False)
    # 'true', 'false', or limit number (e.g., '20', '5')
    # For unlimited: store '-1'
    
    # Optional: Feature scope for future multi-tenant granularity
    # 'tenant' = applies to entire tenant, 'participant' = per participant, 'user' = per user
    feature_scope = db.Column(db.String(50), default='tenant', nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('plan_id', 'feature_key', name='_plan_feature_uc'),)
    
    def __repr__(self):
        return f'<PlanFeature {self.feature_key}={self.feature_value} (Plan {self.plan_id})>'


# --- Direct Messaging System ---

class DirectMessage(db.Model):
    """Direct messages between users (supports multiple recipients)"""
    __tablename__ = 'direct_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    # Sender (single user)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Legacy single recipient (kept for backward compatibility)
    # For new messages with multiple recipients, this will be the first recipient
    recipient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Multi-tenant support (enforce via code: set tenant_id = 1 for single-tenant)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False)
    # Always set tenant_id (even if just 1 for single-tenant) to prevent leaks
    
    # Message context
    message_type = db.Column(db.String(50), default='general')  
    # Values: 'shift_request', 'question', 'general', 'urgent'
    
    # Link to shift (for shift swap requests)
    related_shift_id = db.Column(db.Integer, nullable=True)  # Will be FK to shifts.id when shifts table exists
    
    # Status
    is_urgent = db.Column(db.Boolean, default=False)
    # Note: Read state is tracked via DirectMessageRead table, not is_read field
    # (allows multiple participants and admin views without conflicts)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    recipient = db.relationship('User', foreign_keys=[recipient_id], backref='received_messages')
    recipients = db.relationship('DirectMessageRecipient', backref='message', lazy=True, cascade="all, delete-orphan")
    tenant = db.relationship('Tenant', foreign_keys=[tenant_id])
    replies = db.relationship('DirectMessageReply', backref='message', lazy=True, cascade="all, delete-orphan", order_by='DirectMessageReply.created_at')
    read_by = db.relationship('DirectMessageRead', backref='message', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<DirectMessage {self.id}: {self.subject} (from {self.sender_id})>'
    
    def get_recipients(self):
        """Get all recipients for this message"""
        if self.recipients:
            return [r.recipient for r in self.recipients]
        elif self.recipient_id:
            return [self.recipient]
        return []


class DirectMessageRecipient(db.Model):
    """Junction table for message recipients (supports multiple recipients per message)"""
    __tablename__ = 'direct_message_recipients'
    
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('direct_messages.id'), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ensure a recipient can only appear once per message
    __table_args__ = (db.UniqueConstraint('message_id', 'recipient_id', name='_dm_recipient_uc'),)
    
    # Relationships
    recipient = db.relationship('User', foreign_keys=[recipient_id])
    
    def __repr__(self):
        return f'<DirectMessageRecipient message {self.message_id} to user {self.recipient_id}>'


class DirectMessageReply(db.Model):
    """Replies to direct messages (threaded conversations)"""
    __tablename__ = 'direct_message_replies'
    
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('direct_messages.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def __repr__(self):
        return f'<DirectMessageReply {self.id} to message {self.message_id}>'


class DirectMessageRead(db.Model):
    """Track which users have read which messages"""
    __tablename__ = 'direct_message_reads'
    
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('direct_messages.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    read_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ensure a user can only have one read record per message
    __table_args__ = (db.UniqueConstraint('message_id', 'user_id', name='_dm_user_read_uc'),)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id])
    
    def __repr__(self):
        return f'<DirectMessageRead message {self.message_id} by user {self.user_id}>'


class DirectMessageReplyRead(db.Model):
    """Track which users have read which replies"""
    __tablename__ = 'direct_message_reply_reads'
    
    id = db.Column(db.Integer, primary_key=True)
    reply_id = db.Column(db.Integer, db.ForeignKey('direct_message_replies.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    read_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ensure a user can only have one read record per reply
    __table_args__ = (db.UniqueConstraint('reply_id', 'user_id', name='_dm_reply_user_read_uc'),)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id])
    reply = db.relationship('DirectMessageReply', foreign_keys=[reply_id])
    
    def __repr__(self):
        return f'<DirectMessageReplyRead reply {self.reply_id} by user {self.user_id}>'


# --- Tenant Access Management System ---

class TenantAccess(db.Model):
    """Tracks which organization users can access which tenants"""
    __tablename__ = 'tenant_access'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False)
    organization_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    granted_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # Who granted access
    granted_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # Can be revoked
    notes = db.Column(db.Text, nullable=True)  # Optional notes
    
    # Relationships
    tenant = db.relationship('Tenant', backref='access_grants')
    organization_user = db.relationship('User', foreign_keys=[organization_user_id], backref='tenant_access')
    granted_by = db.relationship('User', foreign_keys=[granted_by_id])
    
    __table_args__ = (db.UniqueConstraint('tenant_id', 'organization_user_id', name='_tenant_org_user_uc'),)
    
    def __repr__(self):
        return f'<TenantAccess tenant {self.tenant_id} -> org_user {self.organization_user_id}>'


class TenantInvitation(db.Model):
    """Invitation links/codes that tenant admins can generate"""
    __tablename__ = 'tenant_invitations'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False)
    invitation_code = db.Column(db.String(50), unique=True, nullable=False)  # e.g., "TAMARA-ABC123"
    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)  # Optional expiry
    max_uses = db.Column(db.Integer, default=1)  # How many times it can be used
    used_count = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text, nullable=True)  # e.g., "For ABC Care Services"
    
    # Relationships
    tenant = db.relationship('Tenant', backref='invitations')
    created_by = db.relationship('User', backref='created_invitations')
    
    def is_valid(self):
        """Check if invitation is still valid"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        if self.used_count >= self.max_uses:
            return False
        return True
    
    def __repr__(self):
        return f'<TenantInvitation {self.invitation_code} for tenant {self.tenant_id}>'


class TenantAccessRequest(db.Model):
    """Requests from organization users to access a tenant"""
    __tablename__ = 'tenant_access_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False)
    requested_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'approved', 'denied'
    reviewed_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)  # Request reason or denial reason
    
    # Relationships
    tenant = db.relationship('Tenant', backref='access_requests')
    requested_by = db.relationship('User', foreign_keys=[requested_by_id], backref='tenant_access_requests')
    reviewed_by = db.relationship('User', foreign_keys=[reviewed_by_id])
    
    def __repr__(self):
        return f'<TenantAccessRequest {self.id}: tenant {self.tenant_id} by user {self.requested_by_id} ({self.status})>'


# --- Booking/Scheduling System (Replaces Setmore) ---

class Service(db.Model):
    """Services that can be booked (e.g., '30 Min Support', '1 hr Support')"""
    __tablename__ = 'services'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False, default=1)
    name = db.Column(db.String(200), nullable=False)  # "30 Min Support", "1 hr Support"
    duration_minutes = db.Column(db.Integer, nullable=False)  # 30, 60, 90, etc.
    category = db.Column(db.String(100))  # "Support", "Professional", "Activity"
    description = db.Column(db.Text)
    price = db.Column(db.Float, default=0.0)  # For future NDIS billing
    is_active = db.Column(db.Boolean, default=True)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    tenant = db.relationship('Tenant', backref='services')
    shifts = db.relationship('Shift', backref='service', lazy=True)
    
    def __repr__(self):
        return f'<Service {self.name} ({self.duration_minutes} min)>'


class Schedule(db.Model):
    """Schedules/Resources (e.g., 'Purple Room Evening', 'Orange Room Daytime')"""
    __tablename__ = 'schedules'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False, default=1)
    name = db.Column(db.String(200), nullable=False)  # "Purple Room Evening"
    short_code = db.Column(db.String(10))  # "PR", "OR", "GR"
    schedule_type = db.Column(db.String(50))  # 'daytime', 'evening', 'weekend_am', 'weekend_pm'
    location = db.Column(db.String(200))  # "Purple Room", "Orange Room"
    color_code = db.Column(db.String(20))  # For calendar display
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    tenant = db.relationship('Tenant', backref='schedules')
    shifts = db.relationship('Shift', backref='schedule', lazy=True)
    
    def __repr__(self):
        return f'<Schedule {self.name}>'


class Shift(db.Model):
    """Scheduled shifts/appointments"""
    __tablename__ = 'shifts'
    
    id = db.Column(db.Integer, primary_key=True)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=False, default=1)
    
    # Service/Activity link
    service_id = db.Column(db.Integer, db.ForeignKey('services.id'), nullable=True)
    
    # Appointment type
    appointment_type = db.Column(db.String(50), default='service')  
    # Values: 'service', 'class', 'event', 'support', 'professional', 'facility_closure'
    
    # Basic info
    title = db.Column(db.String(200), nullable=False)
    
    # Timing
    start_datetime = db.Column(db.DateTime, nullable=False)
    end_datetime = db.Column(db.DateTime, nullable=False)
    duration_hours = db.Column(db.Float)  # Calculated or explicit
    
    # Recurrence
    is_recurring = db.Column(db.Boolean, default=False)
    recurrence_pattern = db.Column(db.String(100))  # 'daily', 'weekly', 'monthly'
    recurrence_end_date = db.Column(db.Date, nullable=True)
    
    # Assignment
    assigned_carer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Schedule/Resource
    schedule_id = db.Column(db.Integer, db.ForeignKey('schedules.id'), nullable=True)
    
    # Location
    location = db.Column(db.String(200))
    
    # Professional appointments
    professional_type = db.Column(db.String(100), nullable=True)  # 'OT', 'Physio', etc.
    professional_name = db.Column(db.String(200), nullable=True)
    
    # Facility closures
    facility_name = db.Column(db.String(200), nullable=True)
    is_closed = db.Column(db.Boolean, default=False)
    
    # Guests/Participants
    guest_count = db.Column(db.Integer, default=1)
    
    # Details
    video_link = db.Column(db.String(500))  # Zoom, Google Meet URL
    notes = db.Column(db.Text)
    label = db.Column(db.String(50), nullable=True)
    color_code = db.Column(db.String(20))
    
    # Booking
    booking_source = db.Column(db.String(100), default='web')  # 'web', 'android_app', 'ios_app', 'admin', 'setmore_import'
    booking_reference = db.Column(db.String(50), unique=True, nullable=True)  # "ST3709", "ST3716"
    
    # Payment tracking (for NDIS billing)
    payment_status = db.Column(db.String(50), default='pending')  # 'pending', 'paid', 'invoiced'
    payment_amount = db.Column(db.Float, nullable=True)
    payment_date = db.Column(db.Date, nullable=True)
    
    # Status
    status = db.Column(db.String(50), default='scheduled')  
    # Values: 'scheduled', 'confirmed', 'cancelled', 'completed'
    
    # Mobile-specific
    is_completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    completed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Integration
    google_event_id = db.Column(db.String(255), nullable=True)  # For Google Calendar sync
    calendar_event_id = db.Column(db.Integer, db.ForeignKey('calendar_events.id'), nullable=True)
    
    # Audit
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tenant = db.relationship('Tenant', backref='shifts')
    assigned_carer = db.relationship('User', foreign_keys=[assigned_carer_id], backref='assigned_shifts')
    creator = db.relationship('User', foreign_keys=[created_by], backref='created_shifts')
    completed_by_user = db.relationship('User', foreign_keys=[completed_by], backref='completed_shifts')
    calendar_event = db.relationship('CalendarEvent', backref='shifts')
    guests = db.relationship('ShiftGuest', backref='shift', lazy=True, cascade="all, delete-orphan")
    reminders = db.relationship('ShiftReminder', backref='shift', lazy=True, cascade="all, delete-orphan")
    history = db.relationship('ShiftHistory', backref='shift', lazy=True, cascade="all, delete-orphan", order_by='ShiftHistory.created_at.desc()')
    payments = db.relationship('ShiftPayment', backref='shift', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Shift {self.title} - {self.start_datetime}>'


class ShiftGuest(db.Model):
    """Guests/participants for a shift"""
    __tablename__ = 'shift_guests'
    
    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=False)
    name = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    is_primary = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<ShiftGuest {self.name} for shift {self.shift_id}>'


class ShiftSwapRequest(db.Model):
    """Request to swap shifts between carers"""
    __tablename__ = 'shift_swap_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=False)
    
    # Who wants to swap
    requester_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    # Who they want to swap with (optional - can be open request)
    target_carer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Their proposed replacement shift (if they have one)
    proposed_shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=True)
    
    status = db.Column(db.String(50), default='pending')  
    # Values: 'pending', 'approved', 'rejected', 'cancelled'
    
    message = db.Column(db.Text)  # Optional message
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    responded_at = db.Column(db.DateTime, nullable=True)
    responded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    requester = db.relationship('User', foreign_keys=[requester_id], backref='swap_requests_sent')
    target_carer = db.relationship('User', foreign_keys=[target_carer_id], backref='swap_requests_received')
    shift = db.relationship('Shift', foreign_keys=[shift_id], backref=db.backref('swap_requests', lazy=True, cascade="all, delete-orphan"))
    proposed_shift = db.relationship('Shift', foreign_keys=[proposed_shift_id])
    responder = db.relationship('User', foreign_keys=[responded_by])
    
    def __repr__(self):
        return f'<ShiftSwapRequest shift {self.shift_id} by user {self.requester_id} ({self.status})>'


class ShiftReminder(db.Model):
    """Reminder settings for shifts"""
    __tablename__ = 'shift_reminders'
    
    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=False)
    reminder_type = db.Column(db.String(50))  # 'email', 'sms', 'push'
    reminder_minutes_before = db.Column(db.Integer)  # 15, 30, 60, etc.
    is_enabled = db.Column(db.Boolean, default=True)
    sent_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<ShiftReminder {self.reminder_type} {self.reminder_minutes_before}min before shift {self.shift_id}>'


class ShiftHistory(db.Model):
    """Audit trail for shift changes"""
    __tablename__ = 'shift_history'
    
    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=False)
    action = db.Column(db.String(50))  # 'created', 'updated', 'cancelled', 'assigned', 'completed'
    changed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    changes = db.Column(db.Text)  # JSON of what changed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', foreign_keys=[changed_by])
    
    def __repr__(self):
        return f'<ShiftHistory shift {self.shift_id} - {self.action} by user {self.changed_by}>'


class ShiftPayment(db.Model):
    """Track payments for shifts (NDIS billing)"""
    __tablename__ = 'shift_payments'
    
    id = db.Column(db.Integer, primary_key=True)
    shift_id = db.Column(db.Integer, db.ForeignKey('shifts.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    payment_date = db.Column(db.Date, nullable=False)
    payment_status = db.Column(db.String(50), default='paid')  # 'paid', 'pending', 'invoiced'
    invoice_number = db.Column(db.String(100), nullable=True)
    ndis_claim_number = db.Column(db.String(100), nullable=True)  # For NDIS billing
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ShiftPayment ${self.amount} for shift {self.shift_id}>'


# --- Admin-Managed Activities System ---

class Activity(db.Model):
    """Activities that can be logged (e.g., Mood, Sleep Quality, Toilet Tries)"""
    __tablename__ = 'activities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False)  # e.g., "Mood", "Sleep Quality"
    display_order = db.Column(db.Integer, default=0)  # Order in dropdown
    is_active = db.Column(db.Boolean, default=True)
    value_type = db.Column(db.String(50), default='dropdown')  # 'dropdown', 'text', 'multi-select'
    # 'dropdown' = single select dropdown
    # 'text' = free text input
    # 'multi-select' = multiple selection (e.g., Exercise, Food)
    
    # For special handling (e.g., Sleep Quality needs different duration dropdown)
    special_duration = db.Column(db.String(50), nullable=True)  # 'sleep', 'nap', None
    
    # For Quick Find search
    synonyms = db.Column(db.Text, nullable=True)  # Comma-separated synonyms
    
    # Multi-tenant support (for future)
    tenant_id = db.Column(db.Integer, db.ForeignKey('tenants.id'), nullable=True)
    # For single-tenant: set to 1 or NULL
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    values = db.relationship('ActivityValue', backref='activity', lazy=True, cascade="all, delete-orphan", order_by='ActivityValue.display_order')
    quick_picks = db.relationship('QuickPickButton', backref='activity', lazy=True, cascade="all, delete-orphan", order_by='QuickPickButton.display_order')
    
    def __repr__(self):
        return f'<Activity {self.name}>'


class ActivityValue(db.Model):
    """Values/options for each activity (e.g., Mood: Upset, Low, Okay, Cheerful, Elated)"""
    __tablename__ = 'activity_values'
    
    id = db.Column(db.Integer, primary_key=True)
    activity_id = db.Column(db.Integer, db.ForeignKey('activities.id'), nullable=False)
    label = db.Column(db.String(200), nullable=False)  # Display text (e.g., "Upset", "Yes Both")
    value = db.Column(db.String(100), nullable=False)  # Stored value (e.g., "1", "4")
    description = db.Column(db.Text, nullable=True)  # Optional longer description
    display_order = db.Column(db.Integer, default=0)  # Order in dropdown
    
    # For Quick Find search
    synonyms = db.Column(db.Text, nullable=True)  # Comma-separated synonyms
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<ActivityValue {self.label} (value={self.value})>'


class QuickPickButton(db.Model):
    """Quick pick buttons that appear on the log entry form"""
    __tablename__ = 'quick_pick_buttons'
    
    id = db.Column(db.Integer, primary_key=True)
    activity_id = db.Column(db.Integer, db.ForeignKey('activities.id'), nullable=False)
    activity_value_id = db.Column(db.Integer, db.ForeignKey('activity_values.id'), nullable=True)
    # If activity_value_id is set, use that value. Otherwise, use manual values below.
    
    # Manual values (used if activity_value_id is NULL)
    value = db.Column(db.String(100), nullable=True)  # Stored value
    value_type = db.Column(db.String(200), nullable=True)  # Display label (e.g., "Tried No Success")
    
    # Button display
    button_text = db.Column(db.String(100), nullable=False)  # e.g., "🚽 Toilet Try"
    display_order = db.Column(db.Integer, default=0)  # Order of buttons
    
    is_active = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    activity_value = db.relationship('ActivityValue', foreign_keys=[activity_value_id])
    
    def __repr__(self):
        return f'<QuickPickButton {self.button_text}>'