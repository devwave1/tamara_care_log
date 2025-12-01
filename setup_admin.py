from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.abspath('instance/care_log.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define the User model (must match models.py)
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()  # Ensure table exists
    # Insert admin user
    password = 'yourpassword'  # Replace with your desired password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')  # Corrected encoding
    if not User.query.filter_by(username='tomf').first():
        new_user = User(username='tomf', email='tomf@example.com', password_hash=hashed_password, is_admin=True, is_approved=True)
        db.session.add(new_user)
        db.session.commit()
    print("Admin user 'tomf' set up or already exists.")