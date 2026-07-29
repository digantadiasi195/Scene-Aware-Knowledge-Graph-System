# scene_classifier.py
import cv2
import torch
import open_clip
from PIL import Image

CLIP_TO_GRAPH_SCENE = {
    "industrial factory scene": "SC2", "kitchen scene": "SC2", "warehouse scene": "SC2",
    "office scene": "SC1", "street scene": "SC1", "laboratory scene": "SC1",
    "living room scene": "SC3", "bedroom scene": "SC4", "bathroom scene": "SC5"
}

class SceneClassifier:
    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model.to(device).eval()
        self.classes = list(CLIP_TO_GRAPH_SCENE.keys())
        tokens = open_clip.tokenize(self.classes).to(device)
        with torch.no_grad():
            self.text_feats = self.model.encode_text(tokens)
            self.text_feats /= self.text_feats.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def predict(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_t = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_t)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ self.text_feats.T).softmax(dim=-1)
        idx = sim.argmax().item()
        return CLIP_TO_GRAPH_SCENE.get(self.classes[idx], "SC1")