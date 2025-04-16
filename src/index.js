import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Create a root using the new createRoot API
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);

// Render the app within the root
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);