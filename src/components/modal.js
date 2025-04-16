import React from 'react';
import './Modal.css';

function Modal({ isOpen, onClose, classification, actionNeeded }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>Rainfall Information</h2>
        <p><strong>Classification:</strong> {classification}</p>
        <p><strong>Action Needed:</strong> {actionNeeded}</p>
        <p>Refer to the prediction below for appropriate measures.</p>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export default Modal;