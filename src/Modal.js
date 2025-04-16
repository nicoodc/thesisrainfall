import React, { useEffect } from 'react';
import './Modal.css';

function Modal({ isOpen, onClose, classification, actionNeeded }) {
  useEffect(() => {
    if (isOpen) {
      // Ensure the user has interacted before playing
      const playSound = () => {
        const audio = new Audio('/test.mp3'); // Path to your audio file
        audio.play().catch(error => console.error("Audio play failed:", error));
        document.removeEventListener('click', playSound); // Remove listener after playing
      };

      document.addEventListener('click', playSound); // Wait for user interaction
    }
  }, [isOpen]);

  if (!isOpen) return null;

  let danger_level = "";

  if (classification === "No Rain") {
    danger_level = "#ADD8E6";
  } else if (classification === "Light Rain") {
    danger_level = "#90EE90";
  } else if (classification === "Moderate Rain") {
    danger_level = "#FFFFE0";
  } else if (classification === "Yellow Rainfall Advisory") {
    danger_level = "#FFFF00";
  } else if (classification === "Orange Rainfall Warning") {
    danger_level = "#FFA500";
  } else if (classification === "Red Rainfall Advisory") {
    danger_level = "#FF0000";
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>Rainfall Information</h2>
        <div><strong>Classification:</strong> <p style={{ color: `${danger_level}` }}>{classification}</p></div>
        <p><strong>Action Needed:</strong> {actionNeeded}</p>
        <p>Refer to the prediction below for appropriate measures.</p>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default Modal;
