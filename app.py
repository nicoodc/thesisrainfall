from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import requests

# Flask app configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = "donsky!"  # Replace with a strong secret key for production
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Replace with the actual IP address of your Raspberry Pi
RPI_API_URL = "https://rainfall-nikkilope.as2.pitunnel.net"

# Function to classify rainfall and flood risk
def classify_rainfall_and_flood(rainfall_mm):
    if rainfall_mm == 0:
        return "No Rain", "No immediate action necessary.", "No action needed"
    elif 0.01 <= rainfall_mm < 2.5:
        return "Light Rain", "No immediate action necessary.", "No action needed"
    elif 2.5 <= rainfall_mm < 7.5:
        return "Moderate Rain", "No flood warning, but stay informed of changing conditions.", "Stay informed"
    elif 7.5 <= rainfall_mm < 15:
        return "Yellow Rainfall Advisory", "Monitor the weather and be warned of potential flooding in low-lying areas.", "Monitor weather"
    elif 15 <= rainfall_mm < 30:
        return "Orange Rainfall Warning", "Be cautious about the possibility of flooding.", "Be cautious"
    elif rainfall_mm >= 30:
        return "Red Rainfall Warning", "Emergency: Heavy rains detected; possibility of severe flooding.", "Emergency actions needed"
    else:
        return "Unknown", "No classification available", "No action available"

@app.route('/')
def index():
    return "Welcome to the Rainfall Monitoring API"

@app.route('/latest_readings', methods=['GET'])
def latest_readings():
    try:
        response = requests.get(f"{RPI_API_URL}/latest_readings")
        if response.status_code == 200:
            data = response.json()
            # Add classification and action logic here
            classification, flood_risk, action_needed = classify_rainfall_and_flood(data["rainfall_15min"])
            data["classification"] = classification
            data["flood_risk"] = flood_risk
            data["action"] = action_needed
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to fetch latest readings"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/last_10_readings', methods=['GET'])
def last_10_readings():
    try:
        response = requests.get(f"{RPI_API_URL}/last_10_readings")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to fetch last 10 readings"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)