#action_recognizer.py
import cv2
import torch
from PIL import Image
from collections import deque # <-- ADD THIS
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


VIDEOMAE_TO_GRAPH_ACTION = {
    "drinking": "a6", "eating": "a6", "reaching": "a2", "stretching arm": "a2",
    "pressing": "a3", "typing": "a3", "pouring": "a4", "opening": "a7"
}

class ActionRecognizer:
    def __init__(self, device):
        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(device).eval()
        
        # FIX: Use deque to automatically drop old frames and prevent memory leaks
        self.buffer = deque(maxlen=16) 

    @torch.inference_mode()
    def predict(self, frame_bgr):
        self.buffer.append(frame_bgr)
        if len(self.buffer) < 16: 
            return "a2" # Default to reaching while buffer fills
        
        # FIX: Convert deque to list for the processor
        selected = list(self.buffer) 
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in selected]
        
        inputs = self.processor(pil_frames, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        action_name = self.model.config.id2label[outputs.logits.argmax(dim=-1).item()].lower()
        
        return VIDEOMAE_TO_GRAPH_ACTION.get(action_name, "a2")


# # perception/action_recognizer.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import deque
# import numpy as np
# import os

# # Mapping from MS-TCN output indices to your Graph Action IDs
# ACTION_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]

# class DilatedResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
#         super(DilatedResidualBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
#         self.relu = nn.ReLU()
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         residual = self.downsample(x)
#         out = self.relu(self.conv1(x))
#         out = self.conv2(out)
#         return self.relu(out + residual)

# class LightweightMSTCN(nn.Module):
#     """
#     A lightweight 2-Stage Temporal Convolutional Network for frame-level 
#     action classification on skeleton sequences.
#     """
#     def __init__(self, num_features=9, num_classes=9, hidden_dim=64):
#         super(LightweightMSTCN, self).__init__()
        
#         # Stage 1: Receptive field expansion
#         self.stage1 = nn.Sequential(
#             nn.Conv1d(num_features, hidden_dim, 1),
#             DilatedResidualBlock(hidden_dim, hidden_dim, dilation=1),
#             DilatedResidualBlock(hidden_dim, hidden_dim, dilation=2),
#             DilatedResidualBlock(hidden_dim, hidden_dim, dilation=4),
#         )
        
#         # Stage 2: Refinement
#         self.stage2 = nn.Sequential(
#             nn.Conv1d(hidden_dim, hidden_dim, 1),
#             DilatedResidualBlock(hidden_dim, hidden_dim, dilation=1),
#             DilatedResidualBlock(hidden_dim, hidden_dim, dilation=2),
#         )
        
#         self.classifier = nn.Conv1d(hidden_dim, num_classes, 1)

#     def forward(self, x):
#         # x shape: (Batch, Features, Time)
#         out = self.stage1(x)
#         out = self.stage2(out)
#         logits = self.classifier(out) # (Batch, Classes, Time)
        
#         # We want the prediction for the *current* (last) frame in the sequence
#         return logits[:, :, -1] # (Batch, Classes)


# class ActionRecognizer:
#     def __init__(self, device, seq_len=16):
#         self.device = device
#         self.num_features = 9  # 3 joints * 3 coords
#         self.num_classes = 9
#         self.seq_len = seq_len
        
#         # Initialize MS-TCN
#         self.model = LightweightMSTCN(
#             num_features=self.num_features, 
#             num_classes=self.num_classes, 
#             hidden_dim=64
#         ).to(device).eval()
        
#         # Try to load pre-trained weights (you will train this with train_gnn.py or a separate script)
#         model_path = "reasoning/ms_tcn_skeleton.pth"
#         if os.path.exists(model_path):
#             self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#             print("[INFO] Loaded pre-trained MS-TCN weights.")
#         else:
#             print("[WARNING] No pre-trained MS-TCN weights found. Using random init. Train it for best results!")

#         # Buffer to store sequential skeleton features
#         self.buffer = deque(maxlen=self.seq_len)

#     @torch.inference_mode()
#     def predict(self, skeleton_features_1d):
#         """
#         Args:
#             skeleton_features_1d: list of 9 floats [rx, ry, rz, lx, ly, lz, cx, cy, cz]
#         Returns:
#             action_id (str): e.g., "a2"
#         """
#         if skeleton_features_1d is None:
#             return "a1" # Default to idle if no skeleton detected

#         # 1. Add new frame features to buffer
#         self.buffer.append(skeleton_features_1d)
        
#         # 2. Pad if buffer isn't full yet
#         if len(self.buffer) < self.seq_len:
#             padded_buffer = list(self.buffer) + [skeleton_features_1d] * (self.seq_len - len(self.buffer))
#         else:
#             padded_buffer = list(self.buffer)
            
#         # 3. Convert to tensor: Shape (Features, Time) -> (1, Features, Time) for batch dim
#         seq_array = np.array(padded_buffer).T # Transpose to (9, 16)
#         x = torch.tensor(seq_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        
#         # 4. Forward pass
#         logits = self.model(x) # (1, 9)
#         probs = F.softmax(logits, dim=-1)
        
#         # 5. Get prediction
#         pred_idx = probs.argmax(dim=-1).item()
#         confidence = probs.max().item()
        
#         # Fallback: If confidence is very low, default to "a2" (reaching/interacting)
#         if confidence < 0.35:
#             return "a2"
            
#         return ACTION_NAMES[pred_idx]