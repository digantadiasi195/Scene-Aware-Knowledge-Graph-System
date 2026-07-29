import React, { useState, useEffect } from 'react';
import { X, Database, Ruler, Zap } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function GraphNodeInspector({ node, onClose }) {
  const [affordances, setAffordances] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!node?.data?.object_id || node.data.type !== 'object') {
      setAffordances(null);
      return;
    }

    let cancelled = false;
    setLoading(true);

    fetch(`${API_URL}/api/scene-options`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) {
          const obj = data.objects.find((o) => o.object_id === node.data.object_id);
          setAffordances(obj || null);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAffordances(null);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [node]);

  if (!node) return null;

  const isHuman = node.data.type === 'human';
  const isScene = node.data.type === 'scene';
  const isObject = node.data.type === 'object';

  return (
    <div className="inspector-overlay" onClick={onClose}>
      <div className="inspector-panel" onClick={(e) => e.stopPropagation()}>
        <div className="inspector-header">
          <div className="inspector-title">
            {isHuman && '👤'}{isObject && ''}{isScene && '🏢'} Node Inspector
          </div>
          <button className="inspector-close" onClick={onClose}><X size={16} /></button>
        </div>

        <div className="inspector-body">
          <div className="inspector-section">
            <div className="inspector-label">Label</div>
            <div className="inspector-value">{node.data.label}</div>
          </div>

          {isObject && affordances && (
            <>
              <div className="inspector-section">
                <div className="inspector-label"><Database size={12} /> Object ID</div>
                <div className="inspector-value mono">{affordances.object_id}</div>
              </div>

              <div className="inspector-section">
                <div className="inspector-label"><Ruler size={12} /> Geometry Priors</div>
                <div className="inspector-grid-3">
                  <div className="inspector-mini-card">
                    <span>d_Ho</span>
                    <strong>{affordances.d_Ho.toFixed(2)}m</strong>
                  </div>
                  <div className="inspector-mini-card">
                    <span>θ</span>
                    <strong>{affordances.theta.toFixed(2)}rad</strong>
                  </div>
                  <div className="inspector-mini-card">
                    <span>e</span>
                    <strong>{affordances.e.toFixed(2)}</strong>
                  </div>
                </div>
              </div>

              <div className="inspector-section">
                <div className="inspector-label"><Zap size={12} /> Afforded Actions</div>
                <div className="tag-list">
                  {affordances.actions && affordances.actions.length > 0 ? (
                    affordances.actions.map((action, i) => (
                      <span key={i} className="affordance-tag">{action}</span>
                    ))
                  ) : (
                    <div className="empty-text">No affordances found</div>
                  )}
                </div>
              </div>
            </>
          )}

          {isHuman && (
            <div className="inspector-section">
              <div className="inspector-label">Type</div>
              <div className="inspector-value">Detected Human Agent</div>
            </div>
          )}

          {isScene && (
            <div className="inspector-section">
              <div className="inspector-label">Scene ID</div>
              <div className="inspector-value mono">{node.data.label.replace('Scene: ', '')}</div>
            </div>
          )}

          {loading && (
            <div className="inspector-loading">
              <div className="spinner-small"></div> Loading KG metadata...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}