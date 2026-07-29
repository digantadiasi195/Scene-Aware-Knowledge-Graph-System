// frontend/src/LiveCamera.jsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Video, VideoOff, Circle, AlertCircle } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function LiveCamera({ onClose }) {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState('');
  const [detections, setDetections] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const processingRef = useRef(false);
  const intervalRef = useRef(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);
        setError('');
        
        // Start processing frames every 2 seconds
        intervalRef.current = setInterval(processFrame, 2000);
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Unable to access camera. Please grant camera permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsActive(false);
    setDetections(null);
  };

  const processFrame = useCallback(async () => {
    if (processingRef.current || !canvasRef.current || !videoRef.current) return;
    
    processingRef.current = true;
    
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to blob
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
      
      // Send to backend for processing
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      
      const response = await fetch(`${API_URL}/api/process-live-frame/`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.json();
        setDetections(result);
      }
    } catch (err) {
      console.error('Frame processing error:', err);
    } finally {
      processingRef.current = false;
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '800px' }}>
        <div className="modal-header">
          <div className="modal-title">
            <Video size={18} color="#22c55e" />
            <span>Live Camera Surveillance</span>
          </div>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {error && (
            <div className="whatif-error" style={{ marginBottom: '16px' }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div style={{ position: 'relative', width: '100%', borderRadius: '12px', overflow: 'hidden', background: '#000' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            
            {/* Live indicator */}
            <div style={{
              position: 'absolute',
              top: '16px',
              left: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: 'rgba(0, 0, 0, 0.7)',
              padding: '8px 12px',
              borderRadius: '8px',
              color: '#fff',
              fontSize: '0.85rem',
              fontWeight: 600
            }}>
              <div className="live-dot" style={{ animation: 'blink 1s infinite' }}></div>
              LIVE
            </div>

            {/* Stop button */}
            <button
              onClick={stopCamera}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                background: '#ef4444',
                color: '#fff',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontWeight: 600
              }}
            >
              <VideoOff size={16} />
              Stop
            </button>
          </div>

          {/* Real-time detections */}
          {detections && (
            <div style={{ marginTop: '20px', padding: '16px', background: '#1e293b', borderRadius: '12px' }}>
              <h3 style={{ margin: '0 0 12px 0', color: '#22c55e', fontSize: '1.1rem' }}>
                🎯 Real-Time Analysis
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>Scene</div>
                  <div style={{ fontSize: '1rem', fontWeight: 600, color: '#38bdf8' }}>
                    {detections.scene || 'Analyzing...'}
                  </div>
                </div>
                
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>Objects</div>
                  <div style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
                    {detections.objects?.length || 0} detected
                  </div>
                </div>
              </div>

              {detections.predicted_goal && (
                <div style={{ marginTop: '16px', padding: '12px', background: '#0b1120', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>Predicted Goal</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#22d3ee' }}>
                    {detections.predicted_goal.goal || 'Unknown'}
                  </div>
                  {detections.predicted_goal.reasoning && (
                    <div style={{ fontSize: '0.85rem', color: '#cbd5e1', marginTop: '8px', fontStyle: 'italic' }}>
                      "{detections.predicted_goal.reasoning}"
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: '16px', fontSize: '0.8rem', color: '#64748b', textAlign: 'center' }}>
            Processing frames every 2 seconds • Objects and goals updated in real-time
          </div>
        </div>
      </div>
    </div>
  );
}