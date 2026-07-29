// components/SkeletonViewer.jsx
import React, { useRef, useEffect } from 'react';

export default function SkeletonViewer({ frameData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    if (!frameData || !frameData.skeleton) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No skeleton data received', width / 2, height / 2);
      return;
    }

    const humans = frameData.skeleton.humans || [];

    if (humans.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No humans detected in this frame', width / 2, height / 2);
      return;
    }

    // Color palette for different humans
    const humanColors = [
      '#22c55e',  // Green
      '#3b82f6',  // Blue
      '#ef4444',  // Red
      '#a855f7',  // Purple
      '#f59e0b',  // Amber
      '#06b6d4',  // Cyan
    ];

    // Draw each human skeleton
    humans.forEach((human, humanIdx) => {
      const lineColor = humanColors[humanIdx % humanColors.length];
      const joints = human.joints || [];
      const connections = human.connections || [];

      // Draw connections (limbs) with better structure
      connections.forEach((conn) => {
        const [startIdx, endIdx] = conn;
        const start = joints[startIdx];
        const end = joints[endIdx];

        if (start && end) {
          ctx.beginPath();
          ctx.moveTo(start.x * width, start.y * height);
          ctx.lineTo(end.x * width, end.y * height);
          ctx.strokeStyle = lineColor;
          ctx.lineWidth = 3;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.stroke();
        }
      });

      // Draw person label (P1, P2, etc.)
      const noseJoint = joints[0];
      if (noseJoint) {
        ctx.fillStyle = lineColor;
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`P${humanIdx + 1}`, noseJoint.x * width, noseJoint.y * height - 20);
      }
    });

    // Draw info
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Humans detected: ${humans.length}`, 10, 20);

  }, [frameData]);

  return (
    <div style={{ width: '100%', height: '450px', display: 'flex', flexDirection: 'column' }}>
      <canvas
        ref={canvasRef}
        width={400}
        height={420} // ✅ Increased internal height for taller skeleton
        style={{ width: '100%', height: '100%', borderRadius: '8px', background: '#0b1120' }}
      />
      <div style={{
        padding: '8px 12px',
        background: '#1e293b',
        borderRadius: '0 0 8px 8px',
        display: 'flex',
        gap: '16px',
        fontSize: '0.7rem',
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '20px', height: '4px', background: '#22c55e', borderRadius: '2px' }} />
          <span style={{ color: '#94a3b8' }}>P1</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '20px', height: '4px', background: '#3b82f6', borderRadius: '2px' }} />
          <span style={{ color: '#94a3b8' }}>P2</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '20px', height: '4px', background: '#ef4444', borderRadius: '2px' }} />
          <span style={{ color: '#94a3b8' }}>P3</span>
        </div>
      </div>
    </div>
  );
}