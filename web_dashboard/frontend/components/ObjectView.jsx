// components/ObjectView.jsx
import React from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function ObjectView({ frameData }) {
  if (!frameData || !frameData.objects || frameData.objects.length === 0) {
    return (
      <div style={{ 
        display: 'flex', alignItems: 'center', justifyContent: 'center', 
        height: '100%', color: '#94a3b8', fontSize: '0.9rem' 
      }}>
        No objects detected
      </div>
    );
  }

  const objectViewUrl = frameData.object_view_url || '';

  return (
    <div style={{ 
      width: '100%', 
      height: '100%', 
      background: '#0b1120',
      padding: '12px'
    }}>
      {/* ✅ Full Frame Object Segmentation - Only visualization */}
      <div style={{ 
        width: '100%', 
        height: '100%', 
        position: 'relative', 
        borderRadius: '8px', 
        overflow: 'hidden', 
        background: '#000'
      }}>
        {objectViewUrl ? (
          <img 
            src={`${API_URL}${objectViewUrl}`} 
            alt="Object Segmentation" 
            style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
          />
        ) : (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: '100%', 
            color: '#94a3b8' 
          }}>
            Processing objects...
          </div>
        )}
      </div>
      {/* ✅ Object list removed - only visualization remains */}
    </div>
  );
}