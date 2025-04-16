import React from 'react';
import Plot from 'react-plotly.js';

function RainfallGraph({ timeData, rainfallData }) {
  return (
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
        layout={{ title: 'Rainfall Over Time' }}
      />
    </div>
  );
}

export default RainfallGraph;