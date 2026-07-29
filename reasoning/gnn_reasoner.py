# reasoning/gnn_reasoner.py
"""
WHY GNN WHEN THE PAPER USES SIMPLE PROBABILITY?

The original paper's probabilistic approach (Eq. 10) works because:
1. Small, fixed object set (12 objects)
2. Single-scenario evaluation
3. Hand-crafted activity graphs (Fig. 9)

Scene-aware prediction introduces COMBINATORIAL EXPLOSION:
- 5 scenes × 12 objects × 9 actions = 540 scene-object-action combinations
- Cross-scene object sharing (bottle in kitchen AND office) creates ambiguity
- Temporal context now depends on scene (pouring follows reaching in kitchen,
  but pressing follows reaching in office)

The GNN learns RELATIONAL REPRESENTATIONS that:
1. Compress high-dimensional scene-object-action joint space
2. Generalize to unseen (scene, object, action) combinations via graph structure
3. Fuse symbolic knowledge (Neo4j priors) with subsymbolic spatial features
"""
# reasoning/gnn_reasoner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# ============================================================
# JOINT CONFIGURATION
# ============================================================
# Total joints available: 0 to 33 (34 joints)
# Excluded joints: 8, 9, 10, 15, 16, 17, 21, 25, 28, 29, 30, 31, 32, 33
INCLUDED_JOINTS = [
    0, 1, 2, 3, 4, 5, 6, 7, 
    11, 12, 13, 14, 
    18, 19, 20, 
    22, 23, 24, 
    26, 27
]
POSE_DIM = len(INCLUDED_JOINTS) * 3  # 20 joints * 3 coords = 60 dimensions

# Feature layouts (must match graph_data_extractor.py exactly)
OBJECT_CLASSES = [
    "o_glass", "o_bottle", "o_computer", "o_doorknob", "o_milk", 
    "o_bowl", "o_object", "o_book", "o_phone", "o_microwave", 
    "o_table", "o_door"
]
SCENE_CLASSES = ["SC1", "SC2", "SC3", "SC4", "SC5"]

# The maximum dimension across all node types (Human=60, Object=16, Scene=5)
NUM_NODE_FEATURES = POSE_DIM  # 60

class GATReasoner(nn.Module):
    """
    Graph Attention Network (GAT) for scene-aware human action reasoning.
    Updated to handle 20-joint human pose (60 dims) + dynamic N-node graphs.
    """

    def __init__(
        self,
        num_node_features: int = 60,  # <-- CRITICAL FIX: Changed from 16 to 60
        hidden_dim: int = 64,         # <-- Increased slightly for richer pose data
        num_classes: int = 9,
        gat_heads: int = 4
    ):
        super(GATReasoner, self).__init__()

        # Graph Attention Layer 1
        self.conv1 = GATConv(
            in_channels=num_node_features,  # This will now correctly be 60
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False  # Average the attention heads
        )

        # Graph Attention Layer 2
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False
        )

        # Fully connected classification head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        
        # Global mean pooling to convert node embeddings -> single graph embedding
        graph_embed = global_mean_pool(x, batch)
        
        # Classification logits
        logits = self.fc(graph_embed)
        return logits

    @torch.no_grad()
    def predict_from_context(
        self,
        scene_id: str,
        objects_with_priors: list,
        all_joints_dict: dict, # Format: {joint_id: (x, y, z)}
        device: torch.device
    ) -> tuple:
        """
        Live inference method.
        all_joints_dict: Dictionary of ALL 34 joints from your pose estimator.
        """
        node_features = []

        # 1. Human Node (20 Joints -> 60 dims)
        human_feat = [0.0] * POSE_DIM
        for i, joint_id in enumerate(INCLUDED_JOINTS):
            if joint_id in all_joints_dict:
                x, y, z = all_joints_dict[joint_id]
                human_feat[i*3 + 0] = x
                human_feat[i*3 + 1] = y
                human_feat[i*3 + 2] = z
        node_features.append(human_feat)

        # 2. Scene Node (5 dims one-hot + 55 zeros padding = 60 dims)
        scene_feat = [0.0] * POSE_DIM
        if scene_id in SCENE_CLASSES:
            scene_feat[SCENE_CLASSES.index(scene_id)] = 1.0
        node_features.append(scene_feat)

        # 3. Object Nodes (16 dims data + 44 zeros padding = 60 dims)
        for obj in objects_with_priors:
            obj_feat = [0.0] * POSE_DIM
            
            # One-Hot Encoding (Indices 0-11)
            obj_id = obj.get("object_id", "o_object")
            if obj_id in OBJECT_CLASSES:
                obj_feat[OBJECT_CLASSES.index(obj_id)] = 1.0
            
            # XYZ Coordinates (Indices 12-14)
            xyz = obj.get("xyz", (0.0, 0.0, 0.0))
            obj_feat[12] = xyz[0]
            obj_feat[13] = xyz[1]
            obj_feat[14] = xyz[2]
            
            # Symbolic Prior (Index 15)
            obj_feat[15] = obj.get("scene_obj_p", 0.5)
            
            # Indices 16-59 remain 0.0 (Padding)
            node_features.append(obj_feat)

        # 4. Build Tensors
        x = torch.tensor(node_features, dtype=torch.float32, device=device)
        num_nodes = x.size(0)

        # 5. Fully Connected Edges
        row, col = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # 6. Predict
        logits = self.forward(x, edge_index, batch)
        probabilities = F.softmax(logits, dim=-1)

        predicted_idx = probabilities.argmax().item()
        confidence = probabilities.max().item()
        action_id = f"a{predicted_idx + 1}"

        return action_id, confidence