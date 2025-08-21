from flask_sqlalchemy import SQLAlchemy
from flask import Flask



app = Flask(__name__)

# PostgreSQL connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://devrate_user:password123@localhost/devrate'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    metrics = db.Column(db.JSON, nullable=False)   # store metrics as JSON
    recommendations = db.Column(db.JSON, nullable=True)

    def __init__(self, filename, metrics, recommendations):
        self.filename = filename
        self.metrics = metrics
        self.recommendations = recommendations