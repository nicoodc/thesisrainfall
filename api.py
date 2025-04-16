from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import sqlite3

app = Flask(__name__)
# Allow requests from multiple origins
CORS(app, resources={r"/*": {"origins": ["http://192.168.18.47:3003", "http://localhost:3003"]}})
socketio = SocketIO(app, cors_allowed_origins=["http://192.168.18.47:3003", "http://localhost:3003"])

# Database path
DATABASE = '/home/pi/rainfall_data2.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return "Welcome to the Rainfall Monitoring API"

@app.route('/latest_readings', methods=['GET'])
def latest_readings():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the latest reading from the rainfall_log table
    cursor.execute("SELECT * FROM rainfall_log ORDER BY timestamp DESC LIMIT 1")
    latest_reading = cursor.fetchone()
    print("Latest Reading:", latest_reading)  # Debug print

    # Fetch the latest prediction from the rainfall_predictions table
    cursor.execute("SELECT * FROM rainfall_predictions ORDER BY timestamp DESC LIMIT 1")
    latest_prediction = cursor.fetchone()
    print("Latest Prediction:", latest_prediction)  # Debug print

    conn.close()

    if latest_reading:
        print("Latest Reading Data: ", dict(latest_reading))
    else:
        print("No latest reading found.")

    if latest_prediction:
        print("Latest Prediction Data: ", dict(latest_prediction))
    else:
        print("No latest prediction found.")
    
    if latest_reading:
        response = {
            "rainfall_15min": latest_reading["rainfall_15min"],
            "classification": latest_reading["rainfall_classification_15min"],
            "predictions": {
                "1hour": latest_prediction["predicted_1hour"] if latest_prediction and latest_prediction["predicted_1hour"] is not None else "No Prediction Data",
                "classification": latest_prediction["rainfall_classification"] if latest_prediction and latest_prediction["rainfall_classification"] is not None else "No Prediction Data",
                "flood_risk": latest_prediction["flood_risk"] if latest_prediction and latest_prediction["flood_risk"] is not None else "No Prediction Data",
                "action_needed": latest_prediction["action_needed"] if latest_prediction and latest_prediction["action_needed"] is not None else "No Prediction Data"
            }
        }
        return jsonify(response)
    else:
        return jsonify({"error": "No data found"}), 404

@app.route('/last_10_readings', methods=['GET'])
def last_10_readings():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the last 10 readings from the rainfall_log table
    cursor.execute("SELECT timestamp, rainfall_15min FROM rainfall_log ORDER BY timestamp DESC LIMIT 10")
    last_10_readings = cursor.fetchall()
    print("Last 10 Readings:", last_10_readings)  # Debug print

    conn.close()

    if last_10_readings:
        readings = []
        for row in last_10_readings:
            readings.append({"timestamp": row["timestamp"], "rainfall_15min": row["rainfall_15min"]})
        return jsonify(readings)
    else:
        return jsonify({"error": "No data found"}), 404

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('updateSensorData')
def handle_update_sensor_data(msg):
    print('Received sensor data:', msg)
    
    socketio.emit('updateSensorData', msg)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)