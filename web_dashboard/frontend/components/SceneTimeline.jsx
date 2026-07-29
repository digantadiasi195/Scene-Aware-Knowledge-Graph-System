// components/SceneTimeline.jsx
import React, { useMemo } from 'react';

// ✅ Extended color palette + dynamic generation fallback
const PREDEFINED_COLORS = {
  'SC1': '#3b82f6', 'SC2': '#22c55e', 'SC3': '#f59e0b', 
  'SC4': '#ef4444', 'SC5': '#a855f7', 'SC6': '#06b6d4', 'SC7': '#ec4899',
  'SC8': '#14b8a6', 'SC9': '#f97316', 'SC10': '#84cc16',
  'SC11': '#0ea5e9', 'SC12': '#d946ef', 'SC13': '#f43f5e',
  'SC14': '#8b5cf6', 'SC15': '#10b981', 'SC16': '#f59e0b',
};

// ✅ Generate a consistent color for any scene ID using hash
function getSceneColor(sceneId) {
  // Check predefined colors first
  if (PREDEFINED_COLORS[sceneId]) {
    return PREDEFINED_COLORS[sceneId];
  }
  
  // Generate color dynamically for unknown scene IDs
  let hash = 0;
  for (let i = 0; i < sceneId.length; i++) {
    hash = sceneId.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const colors = [
    '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a855f7', 
    '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
    '#0ea5e9', '#d946ef', '#f43f5e', '#8b5cf6', '#10b981',
    '#eab308', '#6366f1', '#fb7185', '#2dd4bf', '#fbbf24'
  ];
  
  return colors[Math.abs(hash) % colors.length];
}

export default function SceneTimeline({ data, totalFrames, currentFrame, onSeek }) {
  const scenes = useMemo(() => {
    if (!data || !data.scene_occurrences || data.scene_occurrences.length === 0) return [];
    return data.scene_occurrences;
  }, [data]);

  if (scenes.length === 0 || !totalFrames) {
    return (
      <div style={{ 
        padding: '15px', 
        background: '#1e293b', 
        borderRadius: '8px', 
        color: '#64748b', 
        textAlign: 'center',
        fontSize: '0.9rem',
        border: '1px dashed #334155'
      }}>
        🎬 Scene Timeline will appear here as the video plays...
      </div>
    );
  }

  const playheadPercent = totalFrames > 0 ? (currentFrame / totalFrames) * 100 : 0;

  return (
    <div style={{ 
      padding: '16px', 
      background: '#1e293b', 
      borderRadius: '8px', 
      border: '1px solid #334155',
      position: 'relative'
    }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        marginBottom: '8px',
        fontSize: '0.8rem',
        color: '#94a3b8',
        fontWeight: '600'
      }}>
        <span>🎞️ Scene Timeline</span>
        <span>{scenes.length} Scenes Detected</span>
      </div>

            {/* Timeline Bar - Full Coverage (No Gaps) */}
      <div style={{ 
        position: 'relative', 
        height: '40px', 
        display: 'flex', 
        borderRadius: '6px', 
        overflow: 'hidden',
        cursor: 'pointer',
        boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3)',
        background: '#0f172a'
      }}>
        {scenes.map((scene, idx) => {
          const startFrame = scene.frame_range[0];
          const endFrame = scene.frame_range[1];
          
          // ✅ Fill gap: extend to the start of next scene (or totalFrames for last scene)
          const nextSceneStart = scenes[idx + 1] 
            ? scenes[idx + 1].frame_range[0] 
            : totalFrames;
          
          const duration = nextSceneStart - startFrame; // ✅ Includes gap
          const widthPercent = (duration / totalFrames) * 100;
          const leftPercent = (startFrame / totalFrames) * 100;
          
          const color = getSceneColor(scene.scene_id);
          
          return (
            <div
              key={scene.occurrence_id}
              onClick={() => onSeek(startFrame)}
              title={`${scene.label} (${scene.scene_id}) - Frame ${startFrame}-${endFrame}`}
              style={{
                position: 'absolute',
                left: `${leftPercent}%`,
                width: `${Math.max(widthPercent, 0.5)}%`,
                height: '100%',
                background: color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'filter 0.2s, transform 0.1s',
                cursor: 'pointer',
                borderRight: idx < scenes.length - 1 ? '1px solid rgba(15, 23, 42, 0.5)' : 'none'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.filter = 'brightness(1.3)';
                e.currentTarget.style.transform = 'scaleY(1.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.filter = 'brightness(1)';
                e.currentTarget.style.transform = 'scaleY(1)';
              }}
            >
              {widthPercent > 3 && (
                <span style={{ 
                  color: '#fff', 
                  fontSize: widthPercent > 8 ? '0.75rem' : '0.65rem',
                  fontWeight: 'bold',
                  textShadow: '0 1px 2px rgba(0,0,0,0.6)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  padding: '0 2px',
                  userSelect: 'none'
                }}>
                  {scene.scene_id}
                </span>
              )}
            </div>
          );
        })}

        {/* Playhead Marker */}
        <div style={{
          position: 'absolute',
          left: `${playheadPercent}%`,
          top: 0,
          bottom: 0,
          width: '3px',
          background: '#fff',
          boxShadow: '0 0 6px rgba(255,255,255,0.9)',
          pointerEvents: 'none',
          transition: 'left 0.1s linear',
          zIndex: 20
        }}>
          <div style={{
            position: 'absolute',
            top: '-6px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '0',
            height: '0',
            borderLeft: '5px solid transparent',
            borderRight: '5px solid transparent',
            borderTop: '6px solid #fff'
          }} />
        </div>
      </div>

      {/* Info */}
      <div style={{ 
        marginTop: '8px', 
        fontSize: '0.75rem', 
        color: '#64748b',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <span>👆 Click any block to jump to that scene</span>
        <span>Current: Frame {currentFrame} / {totalFrames}</span>
      </div>
      </div>
  );
}