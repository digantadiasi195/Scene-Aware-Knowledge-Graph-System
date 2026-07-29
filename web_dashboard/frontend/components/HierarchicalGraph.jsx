// components/HierarchicalGraph.jsx
import React, { useEffect } from 'react';
import { 
  ReactFlow, 
  Background, 
  Controls, 
  useNodesState, 
  useEdgesState,
  MarkerType
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

export default function HierarchicalGraph({ data }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    if (!data || !data.scene_occurrences || data.scene_occurrences.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const newNodes = [];
    const newEdges = [];
    const occurrences = data.scene_occurrences;
    
    const sceneIdColors = {
      'SC1': '#3b82f6', 'SC2': '#22c55e', 'SC3': '#f59e0b', 
      'SC4': '#ef4444', 'SC5': '#a855f7', 'SC6': '#06b6d4', 'SC7': '#ec4899',
    };
    const getColor = (sceneId) => sceneIdColors[sceneId] || '#64748b';

    // 1. Root Node
    const rootId = 'root-context';
    newNodes.push({
      id: rootId,
      type: 'default',
      position: { x: 400, y: 20 },
      style: { 
        background: '#1e293b', color: '#f8fafc', padding: '12px 24px', 
        borderRadius: '12px', fontWeight: 'bold', border: '2px solid #64748b',
        fontSize: '14px'
      },
      data: { label: 'Complete Context' },
    });

    // 2. ✅ EVENLY SPACE SCENES (independent of object count)
    const totalScenes = occurrences.length;
    const sceneSpacing = 250; // Distance between scene centers
    const sceneStartX = 400 - ((totalScenes - 1) * sceneSpacing) / 2;

    occurrences.forEach((occ, occIdx) => {
      const occId = `occ-${occ.occurrence_id}`;
      const sceneX = sceneStartX + occIdx * sceneSpacing;
      const color = getColor(occ.scene_id);

      // Scene node
      newNodes.push({
        id: occId,
        type: 'default',
        position: { x: sceneX, y: 140 },
        style: { 
          background: color, color: '#fff', padding: '10px 16px', 
          borderRadius: '10px', fontWeight: 'bold', border: '2px solid #fff',
          fontSize: '11px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
        },
        data: { label: `${occ.label}\n[${occ.frame_range[0]}-${occ.frame_range[1]}]` },
      });

      // Edge from Root to Scene
      newEdges.push({
        id: `edge-root-${occId}`,
        source: rootId,
        target: occId,
        type: 'bezier',
        animated: true,
        style: { stroke: color, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: color },
      });

      // 3. ✅ CENTER OBJECTS UNDER THEIR PARENT SCENE
      const objects = occ.objects || [];
      const humans = occ.humans || [];
      const allItems = [...objects, ...humans];
      
      const itemWidth = 90;
      const itemSpacing = 15;
      const totalItemsWidth = allItems.length * itemWidth + (allItems.length - 1) * itemSpacing;
      
      // Center objects under the scene
      const itemStartX = sceneX - totalItemsWidth / 2 + itemWidth / 2;

      allItems.forEach((item, itemIdx) => {
        // ✅ FIX: Case-insensitive check for person/human
        const labelLower = (item.label || '').toLowerCase();
        const isHuman = labelLower === 'person' || labelLower === 'human';
    
        const isCrossOcc = (item.appears_in_occurrences || []).length > 1;
        const itemId = `item-${occ.occurrence_id}-${item.id}-${itemIdx}`;
        const itemX = itemStartX + itemIdx * (itemWidth + itemSpacing);

        const bgColor = isHuman ? '#ef4444' : '#10b981'; // Red for humans, Green for objects
        const borderColor = isCrossOcc ? '#fbbf24' : '#fff';
        const borderWidth = isCrossOcc ? 3 : 2;

        newNodes.push({
          id: itemId,
          type: 'default',
          position: { x: itemX, y: 280 },
          style: { 
            background: bgColor, color: '#fff', padding: '6px 10px', 
            borderRadius: '8px', fontWeight: '600',
            border: `${borderWidth}px solid ${borderColor}`,
            fontSize: '10px', boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            width: itemWidth
          },
          data: { label: item.label }, // Keep original label case for display
        });

        // Edge from Scene to Object
        newEdges.push({
          id: `edge-${occId}-${itemId}`,
          source: occId,
          target: itemId,
          type: 'bezier',
          animated: true,
          style: { stroke: isHuman ? '#ef4444' : '#10b981', strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: isHuman ? '#ef4444' : '#10b981' },
        });
      }); // End of allItems forEach
    }); // End of occurrences forEach

    setNodes(newNodes);
    setEdges(newEdges);
  }, [data]);

  if (!data || !data.scene_occurrences || data.scene_occurrences.length === 0) {
    return (
      <div style={{ 
        display: 'flex', alignItems: 'center', justifyContent: 'center', 
        height: '100%', color: '#94a3b8', background: '#0b1120', borderRadius: '8px' 
      }}>
        Waiting for scene data...
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%', background: '#0b1120', borderRadius: '8px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.2}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={20} size={1} />
        <Controls 
          style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} 
          showInteractive={false}
        />
      </ReactFlow>
    </div>
  );
}


// import React, { useEffect, useMemo } from 'react';
// import { 
//   ReactFlow, 
//   Background, 
//   Controls, 
//   useNodesState, 
//   useEdgesState,
//   MarkerType
// } from '@xyflow/react';
// import '@xyflow/react/dist/style.css';

// export default function HierarchicalGraph({ data }) {
//   const [nodes, setNodes, onNodesChange] = useNodesState([]);
//   const [edges, setEdges, onEdgesChange] = useEdgesState([]);

//   // ✅ Memoize the graph building to prevent unnecessary recalculations
//   const graphData = useMemo(() => {
//     if (!data || !data.scene_occurrences || data.scene_occurrences.length === 0) {
//       return { nodes: [], edges: [] };
//     }

//     const newNodes = [];
//     const newEdges = [];
//     const occurrences = data.scene_occurrences;
    
//     // ✅ Limit to last 10 scenes for performance (prevent overwhelming the graph)
//     const recentOccurrences = occurrences.slice(-10);
    
//     const sceneIdColors = {
//       'SC1': '#3b82f6', 'SC2': '#22c55e', 'SC3': '#f59e0b', 
//       'SC4': '#ef4444', 'SC5': '#a855f7', 'SC6': '#06b6d4', 'SC7': '#ec4899',
//     };
//     const getColor = (sceneId) => sceneIdColors[sceneId] || '#64748b';

//     // 1. Root Node
//     const rootId = 'root-context';
//     newNodes.push({
//       id: rootId,
//       type: 'default',
//       position: { x: 400, y: 20 },
//       style: { 
//         background: '#1e293b', color: '#f8fafc', padding: '12px 24px', 
//         borderRadius: '12px', fontWeight: 'bold', border: '2px solid #64748b',
//         fontSize: '14px'
//       },
//       data: { label: 'Complete Context' },
//     });

//     // 2. EVENLY SPACE SCENES
//     const totalScenes = recentOccurrences.length;
//     const sceneSpacing = 250;
//     const sceneStartX = 400 - ((totalScenes - 1) * sceneSpacing) / 2;

//     recentOccurrences.forEach((occ, occIdx) => {
//       const occId = `occ-${occ.occurrence_id}`;
//       const sceneX = sceneStartX + occIdx * sceneSpacing;
//       const color = getColor(occ.scene_id);

//       // Scene node
//       newNodes.push({
//         id: occId,
//         type: 'default',
//         position: { x: sceneX, y: 140 },
//         style: { 
//           background: color, color: '#fff', padding: '10px 16px', 
//           borderRadius: '10px', fontWeight: 'bold', border: '2px solid #fff',
//           fontSize: '11px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
//         },
//         data: { label: `${occ.label}\n[${occ.frame_range[0]}-${occ.frame_range[1]}]` },
//       });

//       // Edge from Root to Scene
//       newEdges.push({
//         id: `edge-root-${occId}`,
//         source: rootId,
//         target: occId,
//         type: 'bezier',
//         animated: false, // ✅ Disable animation for better performance
//         style: { stroke: color, strokeWidth: 2 },
//         markerEnd: { type: MarkerType.ArrowClosed, color: color },
//       });

//       // 3. CENTER OBJECTS UNDER THEIR PARENT SCENE
//       const objects = occ.objects || [];
//       const humans = occ.humans || [];
//       const allItems = [...objects, ...humans];
      
//       // ✅ Limit objects per scene to prevent overcrowding
//       const limitedItems = allItems.slice(0, 8);
      
//       const itemWidth = 90;
//       const itemSpacing = 15;
//       const totalItemsWidth = limitedItems.length * itemWidth + (limitedItems.length - 1) * itemSpacing;
      
//       const itemStartX = sceneX - totalItemsWidth / 2 + itemWidth / 2;

//       limitedItems.forEach((item, itemIdx) => {
//         const labelLower = (item.label || '').toLowerCase();
//         const isHuman = labelLower === 'person' || labelLower === 'human';
    
//         const itemId = `item-${occ.occurrence_id}-${item.id}-${itemIdx}`;
//         const itemX = itemStartX + itemIdx * (itemWidth + itemSpacing);

//         const bgColor = isHuman ? '#ef4444' : '#10b981';

//         newNodes.push({
//           id: itemId,
//           type: 'default',
//           position: { x: itemX, y: 280 },
//           style: { 
//             background: bgColor, color: '#fff', padding: '6px 10px', 
//             borderRadius: '8px', fontWeight: '600',
//             fontSize: '10px', width: itemWidth
//           },
//           data: { label: item.label },
//         });

//         // Edge from Scene to Object
//         newEdges.push({
//           id: `edge-${occId}-${itemId}`,
//           source: occId,
//           target: itemId,
//           type: 'bezier',
//           animated: false, // ✅ Disable animation for better performance
//           style: { stroke: bgColor, strokeWidth: 2 },
//           markerEnd: { type: MarkerType.ArrowClosed, color: bgColor },
//         });
//       });
//     });

//     return { nodes: newNodes, edges: newEdges };
//   }, [data]); // ✅ Only rebuild when data changes

//   // ✅ Update nodes and edges when memoized data changes
//   useEffect(() => {
//     setNodes(graphData.nodes);
//     setEdges(graphData.edges);
//   }, [graphData, setNodes, setEdges]);

//   if (!data || !data.scene_occurrences || data.scene_occurrences.length === 0) {
//     return (
//       <div style={{ 
//         display: 'flex', alignItems: 'center', justifyContent: 'center', 
//         height: '100%', color: '#94a3b8', background: '#0b1120', borderRadius: '8px' 
//       }}>
//         <div style={{ textAlign: 'center' }}>
//           <div style={{ fontSize: '24px', marginBottom: '10px' }}>📊</div>
//           <div>Waiting for scene data...</div>
//           <div style={{ fontSize: '12px', marginTop: '5px', color: '#64748b' }}>
//             Play the video to build the graph
//           </div>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div style={{ width: '100%', height: '100%', background: '#0b1120', borderRadius: '8px' }}>
//       <ReactFlow
//         nodes={nodes}
//         edges={edges}
//         onNodesChange={onNodesChange}
//         onEdgesChange={onEdgesChange}
//         fitView
//         fitViewOptions={{ padding: 0.3 }}
//         minZoom={0.2}
//         maxZoom={1.5}
//         proOptions={{ hideAttribution: true }}
//         defaultEdgeOptions={{ type: 'bezier' }}
//       >
//         <Background color="#1e293b" gap={20} size={1} />
//         <Controls 
//           style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} 
//           showInteractive={false}
//         />
//       </ReactFlow>
//     </div>
//   );
// }