// App.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Upload, Play, Database, Activity, Layers, AlertCircle, 
  Loader, Sparkles, Box, Brain, Video, Eye
} from 'lucide-react';

// ✅ UPDATED IMPORTS
import HierarchicalGraph from './components/HierarchicalGraph';
import SkeletonViewer from './components/SkeletonViewer';
import ObjectView from './components/ObjectView'; // <-- Replaced Scene3DViewer
import SceneTimeline from './components/SceneTimeline';

import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// 🛡️ ERROR BOUNDARY: Prevents blank screen and shows the exact error
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("Frontend UI Crash:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '40px', color: '#ef4444', background: '#0f172a', minHeight: '100vh', fontFamily: 'monospace' }}>
          <h2>⚠️ UI Component Crashed</h2>
          <pre style={{ background: '#1e293b', padding: '20px', color: '#fff', borderRadius: '8px', overflow: 'auto' }}>
            {this.state.error?.toString()}
          </pre>
          <button onClick={() => window.location.reload()} style={{ padding: '10px 20px', marginTop: '20px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function DashboardContent() {
  const [sessionId, setSessionId] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [totalFrames, setTotalFrames] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.5); // ✅ Start at 0.5x speed
  const [frameData, setFrameData] = useState(null);
  const [segmentationUrl, setSegmentationUrl] = useState('');
  const [hierarchicalGraph, setHierarchicalGraph] = useState(null);
  const [videoError, setVideoError] = useState(false);
  const [lastPredictedGoal, setLastPredictedGoal] = useState(null);
  const handleSeek = (frameNum) => {
    if (videoRef.current) {
      const fps = 30; // Assuming 30fps, adjust if your video is different
      const timeInSeconds = frameNum / fps;
      videoRef.current.currentTime = timeInSeconds;
      console.log(`️ Seeking to frame ${frameNum} (${timeInSeconds}s)`);
    }
  };
  const videoRef = useRef(null);
  const processingRef = useRef(false);
  const lastProcessedFrame = useRef(-1);

  const handleFileChange = async (file) => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_URL}/upload-video/`, { method: 'POST', body: formData });
      const data = await res.json();
      
      setSessionId(data.session_id);
      setTotalFrames(data.total_frames);
      setVideoUrl(`${API_URL}${data.video_url}`);
      setFrameData(null);
      setSegmentationUrl('');
      setHierarchicalGraph(null);
      setLastPredictedGoal(null);
      setVideoError(false);
      setPlaybackSpeed(0.5); // Reset to slow speed for new video
      lastProcessedFrame.current = -1;
      processingRef.current = false;
    } catch (err) {
      console.error("❌ Upload failed:", err);
    }
  };

  const processCurrentFrame = useCallback(async (frameNum) => {
    if (!sessionId || processingRef.current) return;
    
    processingRef.current = true;
    
    try {
      const res = await fetch(`${API_URL}/api/process-frame/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, frame_num: frameNum })
      });
      
      const data = await res.json();
      if (!data.error) {
        setFrameData(data);
        setSegmentationUrl(data.segmentation_url || '');
        
        if (data.predicted_goal && data.predicted_goal.goal !== "Unknown") {
          setLastPredictedGoal(data.predicted_goal);
        }
        
        if (frameNum % 30 === 0) {
          const graphRes = await fetch(`${API_URL}/api/hierarchical-graph/${sessionId}`);
          const graphData = await graphRes.json();
          setHierarchicalGraph(graphData);
        }
      }
    } catch (error) {
      console.error('❌ Processing error:', error);
    } finally {
      processingRef.current = false;
    }
  }, [sessionId]);

  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current || !sessionId) return;
    
    const time = videoRef.current.currentTime;
    const fps = 30; 
    const frameNum = Math.floor(time * fps);
    
    setCurrentFrame(frameNum);
    
    // ✅ INCREASED: Process every 15 frames instead of 10 (less load)
    if (frameNum % 15 === 0 && frameNum !== lastProcessedFrame.current) {
      lastProcessedFrame.current = frameNum;
      processCurrentFrame(frameNum);
    }
  }, [sessionId, processCurrentFrame]);

  const handlePlay = () => {
    setIsPlaying(true);
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackSpeed; // ✅ Apply speed
      videoRef.current.play().catch(e => console.error("Play failed:", e));
    }
  };

  const handlePause = () => {
    setIsPlaying(false);
    if (videoRef.current) {
      videoRef.current.pause();
    }
  };

  const handleSpeedChange = (newSpeed) => {
    setPlaybackSpeed(newSpeed);
    if (videoRef.current && isPlaying) {
      videoRef.current.playbackRate = newSpeed;
    }
  };

  return (
    <div className="dashboard">
      {/* NAVBAR */}
      <div className="navbar">
        <div className="navbar-brand">
          <Layers size={24} color="#646cff" />
          <span>Scene-Aware Human-Centric - Real-Time</span>
        </div>
        <div className="navbar-actions">
          {sessionId && (
            <>
              <span style={{color: '#94a3b8', fontSize: '0.85rem', marginRight: '16px'}}>
                Frame: {currentFrame} / {totalFrames}
              </span>
              {/* ✅ PLAYBACK SPEED CONTROL */}
              <div style={{ display: 'flex', gap: '4px' }}>
                {[0.25, 0.5, 0.75, 1.0].map(speed => (
                  <button
                    key={speed}
                    onClick={() => handleSpeedChange(speed)}
                    style={{
                      background: playbackSpeed === speed ? '#3b82f6' : '#1e293b',
                      color: playbackSpeed === speed ? '#fff' : '#94a3b8',
                      border: '1px solid #334155',
                      padding: '4px 10px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.75rem',
                      fontWeight: playbackSpeed === speed ? 'bold' : 'normal'
                    }}
                  >
                    {speed}x
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="main-layout">
        {/* SIDEBAR */}
        <div className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-title">Video Input</div>
            <div className="upload-zone">
              <input type="file" accept="video/*" onChange={(e) => handleFileChange(e.target.files[0])} id="video-upload" hidden />
              <label htmlFor="video-upload" className="upload-label">
                <Upload size={32} color="#94a3b8" />
                <p>Upload Video</p>
              </label>
            </div>
          </div>

          {sessionId && (
            <div className="sidebar-section">
              <div className="sidebar-title">Playback Controls</div>
              <div className="playback-controls">
                <button className="control-btn" onClick={isPlaying ? handlePause : handlePlay}>
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
                <div style={{ 
                  fontSize: '0.7rem', 
                  color: '#94a3b8', 
                  marginTop: '8px',
                  textAlign: 'center',
                  padding: '4px',
                  background: 'rgba(30, 41, 59, 0.5)',
                  borderRadius: '4px'
                }}>
                  {processingRef.current ? '⏳ Processing...' : '✅ Ready'}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* MAIN CONTENT */}
        <div className="main-content">
          {!sessionId ? (
            <div className="placeholder-screen">
              <div className="placeholder-icon"><Activity size={40} /></div>
              <h2>Upload a Video to Begin</h2>
              <p className="processing-text">
                The system will process frames in real-time as you play the video.<br/>
                <strong>Tip: Start at 0.5x speed for best synchronization.</strong>
              </p>
            </div>
          ) : (
            <>
              {/* ROW 1: Original Video | Segmentation | Object View */}
              <div className="visual-grid-3">
                <div className="card">
                  <div className="card-header">
                    <div className="card-title"><Play size={16} /> Original Video</div>
                    <span className="badge badge-blue">{playbackSpeed}x Speed</span>
                  </div>
                  <div className="video-container">
                    {videoError ? (
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#ef4444', padding: '20px', textAlign: 'center' }}>
                        <AlertCircle size={32} />
                        <p style={{ marginTop: '10px', fontSize: '0.9rem' }}>Browser cannot play this video format.<br/>Please use <strong>MP4 (H.264)</strong> or <strong>WebM</strong>.</p>
                      </div>
                    ) : (
                      <video 
                        key={videoUrl}
                        ref={videoRef}
                        src={videoUrl}
                        controls
                        playbackRate={playbackSpeed}
                        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                        onTimeUpdate={handleTimeUpdate}
                        onPlay={handlePlay}
                        onPause={handlePause}
                        onError={(e) => {
                          console.error("❌ Video element error:", e);
                          setVideoError(true);
                        }}
                      />
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-title"><Box size={16} /> Segmentation View</div>
                    <span className="badge badge-purple">Mask2Former</span>
                  </div>
                  <div className="image-container">
                    {segmentationUrl ? (
                      <img src={`${API_URL}${segmentationUrl}`} alt="Segmentation" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    ) : (
                      <div className="empty-view"><Loader size={20} className="spinner" /><span>Processing...</span></div>
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-title"><Box size={16} /> Object View</div>
                    <span className="badge badge-green">Detected Objects</span>
                  </div>
                  <div className="object-view-container">
                    {frameData ? (
                      <ObjectView frameData={frameData} />
                    ) : (
                      <div className="empty-view"><Loader size={20} className="spinner" /><span>Waiting for data...</span></div>
                    )}
                  </div>
                </div>
              </div>

              {/* ✅ NEW ROW: Interactive Scene Timeline */}
              <div className="card" style={{ marginTop: '20px' }}>
                <SceneTimeline 
                  data={hierarchicalGraph} 
                  totalFrames={totalFrames} 
                  currentFrame={currentFrame}
                  onSeek={handleSeek}
                />
              </div>

              {/* ROW 2: Skeleton | Knowledge Graph */}
              <div className="visual-grid-2">
                <div className="card">
                  <div className="card-header">
                    <div className="card-title"><Activity size={16} /> Skeleton View</div>
                    <span className="badge badge-green">YOLO-Pose</span>
                  </div>
                  <div className="skeleton-container">
                    {frameData ? (
                      <SkeletonViewer frameData={frameData} />
                    ) : (
                      <div className="empty-view"><Loader size={20} className="spinner" /><span>Waiting for pose data...</span></div>
                    )}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-title"><Database size={16} /> Hierarchical Knowledge Graph</div>
                    <span className="badge badge-blue">{hierarchicalGraph?.scenes?.length || 0} Scenes</span>
                  </div>
                  <div className="graph-container-large">
                    {hierarchicalGraph ? (
                      <HierarchicalGraph data={hierarchicalGraph} />
                    ) : (
                      <div className="empty-view"><Loader size={20} className="spinner" /><span>Building graph...</span></div>
                    )}
                  </div>
                </div>
              </div>

              {/* LLM Reasoning Banner */}
              {lastPredictedGoal && lastPredictedGoal.goal && lastPredictedGoal.goal !== "Unknown" && (
                <div className="card llm-banner">
                  <div className="card-header">
                    <div className="card-title"><Brain size={18} /> LLM Cognitive Reasoning</div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <span className="badge badge-purple-soft">{lastPredictedGoal.engine || 'LLM'}</span>
                      <span className="badge badge-cyan">Confidence: {lastPredictedGoal.confidence || 'Medium'}</span>
                    </div>
                  </div>
                  <div className="llm-content">
                    <div>
                      <div className="llm-label">Predicted Goal</div>
                      <div className="llm-goal">{lastPredictedGoal.goal}</div>
                    </div>
                    <div>
                      <div className="llm-label">Why?</div>
                      <div className="llm-reasoning">"{lastPredictedGoal.reasoning}"</div>
                    </div>
                    <div>
                      <div className="llm-label">Robot Action</div>
                      <div className="llm-action">→ {lastPredictedGoal.robot_assist}</div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <DashboardContent />
    </ErrorBoundary>
  );
}