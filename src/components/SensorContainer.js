import React from 'react';

function SensorContainer({ sensorData }) {
  return (
    <div className="sensor-container">
      <div id="rainfall" className="box box-rainfall">
        <div className="box-topic">Rainfall</div>
        <div className="number">{sensorData.rainfall_15min.toFixed(2)} mm</div>
      </div>
      <div id="classification" className="box box-classification">
        <div className="box-topic">Classification</div>
        <div id="classification-content">{sensorData.classification}</div>
      </div>
      <div id="action" className="box box-action">
        <div className="box-topic">Action Needed</div>
        <div id="action-content">{sensorData.action}</div>
      </div>
    </div>
  );
}

export default SensorContainer;