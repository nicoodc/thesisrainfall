import React, { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';
import Plot from 'react-plotly.js';
import Modal from './Modal';  // Import the Modal component
import './App.css';

// Ensure the correct URL for the Socket.IO server
const socket = io("https://green-dog-29.telebit.io");

// Function to get the appropriate CSS class based on the classification
function getClassificationColor(classification) {
  switch (classification) {
    case 'No Rain':
      return '#ADD8E6';
    case 'Light Rain':
      return '#90EE90';
    case 'Moderate Rain':
      return '#FFFFE0';
    case 'Yellow Rainfall Advisory':
      return '#FFFF00';
    case 'Orange Rainfall Warning':
      return '#FFA500';
    case 'Red Rainfall Warning':
      return '#FF0000';
    default:
      return '';
  }
}

function App() {
  const [sensorData, setSensorData] = useState({
    rainfall_15min: 0.0,
    classification: "No Rain",
    flood_risk: "No Data",
    action: "No immediate action necessary.",
    predictions: {
      "1hour": 0.0,
      "classification": "No Rain",
      "flood_risk": "No Data",
      "action_needed": "No immediate action necessary."
    }
  });
  const [timeData, setTimeData] = useState([]);
  const [rainfallData, setRainfallData] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);  // State to control modal visibility

  const previousSensorDataRef = useRef(null);

  useEffect(() => {
    // Function to fetch data
    const fetchInitialData = async () => {
      try {
        const readingsResponse = await fetch('https://green-dog-29.telebit.io/last_10_readings');
        if (!readingsResponse.ok) {
          throw new Error(`HTTP error! status: ${readingsResponse.status}`);
        }
        const readingsData = await readingsResponse.json();
        const reversedData = readingsData.reverse(); // Ensure data is in the correct order
        console.log("Readings Data:", readingsData);  // Debug print
        setTimeData(reversedData.map(reading => new Date(reading.timestamp).toLocaleTimeString()));
        setRainfallData(reversedData.map(reading => reading.rainfall_15min));
  
        const latestResponse = await fetch('https://green-dog-29.telebit.io/latest_readings');
        if (!latestResponse.ok) {
          throw new Error(`HTTP error! status: ${latestResponse.status}`);
        }
        const latestData = await latestResponse.json();
        console.log("Latest Data:", latestData);  // Debug print
        if (!latestData || latestData.rainfall_15min == null) {
          throw new Error("Invalid data format received from API");
        }
        setSensorData(latestData);
  
        // Debugging: Log prediction data
        console.log("Prediction Data:", latestData.predictions);
  
        if (!previousSensorDataRef.current || previousSensorDataRef.current.rainfall_15min !== latestData.rainfall_15min) {
          setSensorData(latestData);
          checkRainfallLevel(latestData.rainfall_15min);
        }

        previousSensorDataRef.current = latestData;
      } catch (error) { 
        console.error('Error fetching initial data:', error);
      }
    };
  
    // Run the fetchInitialData every second
    const intervalId = setInterval(fetchInitialData, 1000); // 1000 ms = 1 second
  
    // WebSocket setup for receiving updates
    socket.on("updateSensorData", (msg) => {
      console.log("Received sensor data:", msg); // Debug log
      setSensorData(msg);
  
      // Debugging: Log prediction data
      console.log("Prediction Data from Socket:", msg.predictions);
  
      const currentTime = new Date().toLocaleTimeString();
      const updatedTimeData = [...timeData, currentTime];
      const updatedRainfallData = [...rainfallData, parseFloat(msg.rainfall_15min.toFixed(2))];
  
      if (updatedTimeData.length > 10) {
        updatedTimeData.shift();
        updatedRainfallData.shift();
      }
  
      setTimeData(updatedTimeData);
      setRainfallData(updatedRainfallData);
  
      // Check rainfall levels and open modal if necessary
      checkRainfallLevel(msg.rainfall_15min);
    });
  
    // Cleanup: Clear interval and remove socket listener when component is unmounted
    return () => {
      clearInterval(intervalId); // Clear the interval when the component unmounts
      socket.off("updateSensorData"); // Cleanup the socket listener
    };
  }, [timeData, rainfallData]);  // Add timeData and rainfallData as dependencies
  

  const closeModal = () => {
    setModalOpen(false);
  };

  const openModal = () => {
    setModalOpen(true);
  };

  const checkRainfallLevel = (rainfall_mm) => {
    if (7.5 <= rainfall_mm && rainfall_mm < 15) {
      setModalOpen(true);
    } else if (15 <= rainfall_mm && rainfall_mm < 30) {
      setModalOpen(true);
    } else if (rainfall_mm >= 30) {
      setModalOpen(true);
    } else {
      closeModal();
    }
  };

  return (
    <div className="container">
      <div className="header">
        <ion-icon name="rainy-outline"></ion-icon>
        <span className="header-text">Rainfall Dashboard</span>
        <ion-icon name="rainy-outline"></ion-icon>
      </div>

      <Modal
        isOpen={modalOpen}
        onClose={closeModal}
        classification={sensorData.classification}
        actionNeeded={sensorData.action}
      />

      <div className="sensor-container">
        <div id="rainfall" className="box box-rainfall">
          <div className="box-topic">Rainfall</div>
          <div className="number">{sensorData.rainfall_15min !== null ? sensorData.rainfall_15min.toFixed(2) : "N/A"} mm</div> {/* Display latest rainfall data */}
        </div>
        <div id="classification" className={`box box-classification}`}>
          <div className="box-topic">Classification</div>
          <div id="classification-content" style={{color: `${getClassificationColor(sensorData.classification)}`}}>{sensorData.classification}</div>
        </div>
        <div id="action" className="box box-action">
          <div className="box-topic">Action Needed</div>
          <div id="action-content">{sensorData.action}</div>
        </div>
      </div>

      <div id="rainfall-graph" className="box box-graph">
        <div className="box-topic">Rainfall Trend</div>
        <Plot
          data={[
            {
              x: timeData,
              y: rainfallData,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: '#1f77b4' },
            },
          ]}
          layout={{ title: 'Rainfall Over Time', autosize: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>

      <div className="sensor-container">
        <div id="prediction-1hr" className="box box-rainfall">
          <div className="box-topic">1 Hour Prediction</div>
          <div className="number">{typeof sensorData.predictions["1hour"] === 'number' ? sensorData.predictions["1hour"].toFixed(2) : "No Prediction Data"} mm</div>
        </div>
        <div id="prediction-classification" className={`box box-classification`}>
          <div className="box-topic">Classification</div>
          <div id="prediction-classification-content" style={{color: `${getClassificationColor(sensorData.predictions.classification)}`}}>{sensorData.predictions.classification}</div>
        </div>
        <div id="prediction-action" className="box box-action">
          <div className="box-topic">Action Needed</div>
          <div id="prediction-action-content">{sensorData.predictions.action_needed}</div>
        </div>
      </div>
    </div>
  );
}

export default App;
