import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
import os
import pickle

class SearchEngine:
    def __init__(self):
        # Using MobileNetV3-Large: senior choice for CPU-optimized inference
        self.model = models.mobilenet_v3_large(weights='DEFAULT').eval()
        self.extractor = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img_path):
        """Extracts high-dimensional features with error safety."""
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = self.extractor(img_t).flatten().numpy()
            # L2 Normalization is critical for accurate similarity scoring
            norm = np.linalg.norm(feat)
            return feat / norm if norm > 0 else feat
        except Exception as e:
            return None

    def create_db(self, folder):
        target_folder = os.path.join(folder, "images")
        if not os.path.exists(target_folder): return None, []
        
        paths = [os.path.join(target_folder, f) for f in os.listdir(target_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        raw_feats = [self.extract(p) for p in paths]
        # Filter out failed extractions
        valid_data = [(f, p) for f, p in zip(raw_feats, paths) if f is not None]
        feats, valid_paths = zip(*valid_data)
        
        index = faiss.IndexFlatIP(len(feats[0]))
        index.add(np.array(feats).astype('float32'))
        
        # Persistence layer
        faiss.write_index(index, "fashion_v1.index")
        with open("paths_v1.pkl", "wb") as f:
            pickle.dump(list(valid_paths), f)
        return index, list(valid_paths)

    def load_db(self):
        if os.path.exists("fashion_v1.index") and os.path.exists("paths_v1.pkl"):
            return faiss.read_index("fashion_v1.index"), pickle.load(open("paths_v1.pkl", "rb"))
        return None, None