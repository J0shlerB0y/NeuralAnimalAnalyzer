import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json
from PIL import Image
import io

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CavyNet(nn.Module):
    def __init__(self, num_classes):
        super(CavyNet, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

class CavyPredictor:
    def __init__(self, model_dir="."):
        model_path = os.path.join(model_dir, "cavy_model.pth")
        mapping_path = os.path.join(model_dir, "class_mapping.json")
        emb_path = os.path.join(model_dir, "database_embeddings.npy")
        paths_path = os.path.join(model_dir, "database_paths.npy")
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError("Файлы модели не найдены!")

        with open(mapping_path, "r") as f:
            self.class_names = json.load(f)

        self.model = CavyNet(num_classes=len(self.class_names)).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True) 
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.db_embeddings = np.load(emb_path)
        self.db_paths = np.load(paths_path)

        self.neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.neigh.fit(self.db_embeddings)
        print("Предиктор готов!")

    def predict_bytes(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_tensor = data_transforms(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, embedding = self.model(img_tensor)

            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_class = self.class_names[pred_idx.item()]

            query_vec = embedding.cpu().numpy()
            distances, indices = self.neigh.kneighbors(query_vec)
            
            raw_path = str(self.db_paths[indices[0][0]])
            
            clean_path = raw_path.replace('\\', '/')
            
            if "dataset" in clean_path:
                idx = clean_path.find("dataset")
                clean_path = clean_path[idx:]
            
            similar_path = clean_path
            found_img_bytes = b""
            try:
                with open(similar_path, "rb") as f:
                    found_img_bytes = f.read()
            except Exception as e:
                print(f"Ошибка чтения файла (исправленный путь: {similar_path}): {e}")

            return {
                "species": predicted_class,
                "confidence": float(conf.item()),
                "path": str(similar_path),
                "similar_bytes": found_img_bytes
            }
        except Exception as e:
            print(f"Ошибка при обработке: {e}")
            return None