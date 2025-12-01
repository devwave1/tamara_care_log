# Tamara Care Log

A comprehensive care logging and analysis system for tracking daily care activities, AAC communication trials, and generating insights.

## Features

- **Daily Care Logging**: Track mood, sleep quality, communication levels, toilet attempts, meltdowns, walking ability, food intake, and more
- **AAC Trial Tracking**: Record and monitor Augmentative and Alternative Communication (AAC) device usage and GAS (Goal Attainment Scaling) goals
- **Media Attachments**: Upload and manage photos/videos associated with care log entries
- **Data Analysis Dashboard**: Generate insights and analyze patterns in care data
- **User Management**: Admin panel for managing carers and user accounts
- **Google Sheets Integration**: Sync data to Google Sheets (optional)

## Project Structure

```
tamara_care_log/
├── app.py              # Main Flask application
├── models.py           # Database models
├── utils.py            # Utility functions
├── requirements.txt     # Python dependencies
├── templates/          # HTML templates
├── static/             # CSS, JS, images, uploads
└── instance/           # Database files (NOT in Git)
```

## Setup

### Prerequisites
- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/devwave1/tamara_care_log.git
cd tamara_care_log
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python app.py
```
This will create the database and seed initial data.

4. Create admin user (optional):
```bash
python setup_admin.py
```

5. Run the application:
```bash
python app.py
```

6. Access the application:
- Open your browser to: `http://localhost:5001`

## Configuration

- **Database**: SQLite (`instance/care_log_v1.sqlite`)
- **Port**: 5001
- **Upload Folder**: `static/uploads_test/`
- **Max Upload Size**: 100MB

## Important Notes

⚠️ **Data Privacy**: 
- Database files (`*.sqlite`) are NOT committed to Git (they contain private care data)
- Uploaded media files are NOT committed to Git
- Always backup your database separately

## Technologies Used

- Flask - Web framework
- SQLAlchemy - Database ORM
- Bootstrap 5 - UI framework
- TinyMCE - Rich text editor
- Chart.js - Data visualization

## License

Private project - All rights reserved

## Contact

For questions or support, contact the project maintainer.

