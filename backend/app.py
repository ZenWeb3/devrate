from flask import Flask
from flask_cors import CORS  # import CORS
from backend.routes.analyze import predict_bp

app = Flask(__name__)
CORS(app)  # enable CORS for all origins; you can restrict to your frontend later
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    app.run(debug=True)
