import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {DEVICE}")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CavyNet(nn.Module):
    def __init__(self, num_classes):
        super(CavyNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Забираем количество входных признаков перед последним слоем (обычно 2048)
        num_features = self.backbone.fc.in_features
        
        # Убираем последний слой (fc), чтобы получать эмбеддинги
        # Мы заменяем self.backbone.fc на Identity (пустышку), 
        # чтобы форвард пасс возвращал векторы 2048
        self.backbone.fc = nn.Identity()
        
        # Создаем свой классификатор
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Получаем эмбеддинг (вектор признаков)
        embeddings = self.backbone(x)
        # Получаем предсказание класса
        logits = self.classifier(embeddings)
        return logits, embeddings

def train_model():
    train_dataset = datasets.ImageFolder("dataset", transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    class_names = train_dataset.classes
    print(f"Классы: {class_names}")
    
    model = CavyNet(num_classes=len(class_names)).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(15):
        running_loss = 0.0
        print(f"start epoch {epoch+1} train")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            logits, _ = model(inputs) # Эмбеддинги при обучении не нужны
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), "cavy_model.pth")
    
    import json
    with open("class_mapping.json", "w") as f:
        json.dump(class_names, f)
        
    print("Модель обучена и сохранена.")

def create_embeddings_database():
    # Загружаем модель
    import json
    with open("class_mapping.json", "r") as f:
        class_names = json.load(f)
        
    model = CavyNet(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load("cavy_model.pth"))
    model.eval()
    
    dataset = datasets.ImageFolder("dataset", transform=data_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_embeddings = []
    all_paths = [s[0] for s in dataset.samples]
    
    print("Генерация эмбеддингов для базы")
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            _, embeddings = model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            
    all_embeddings = np.vstack(all_embeddings)
    
    np.save("database_embeddings.npy", all_embeddings)
    np.save("database_paths.npy", np.array(all_paths))
    print("База эмбеддингов создана.")

class CavyPredictor:
    def __init__(self):
        # Загрузка метаданных
        import json
        with open("class_mapping.json", "r") as f:
            self.class_names = json.load(f)
            
        # Загрузка модели
        self.model = CavyNet(num_classes=len(self.class_names)).to(DEVICE)
        self.model.load_state_dict(torch.load("cavy_model.pth", map_location=DEVICE))
        self.model.eval()
        
        # Загрузка базы для поиска похожих
        self.db_embeddings = np.load("database_embeddings.npy")
        self.db_paths = np.load("database_paths.npy")
        
        # Инициализация поиска ближайших соседей (используем косинусное расстояние или евклидово)
        self.neigh = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.neigh.fit(self.db_embeddings)
        
    def predict(self, image_path):
        # Загрузка и подготовка картинки
        img = Image.open(image_path).convert('RGB')
        img_tensor = data_transforms(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits, embedding = self.model(img_tensor)
            
        # Определение вида
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        predicted_class = self.class_names[pred_idx.item()]
        
        # Поиск похожего
        # embedding нужно перевести в numpy
        query_vec = embedding.cpu().numpy()
        distances, indices = self.neigh.kneighbors(query_vec)
        
        similar_img_path = self.db_paths[indices[0][0]]
        
        return {
            "predicted_species": predicted_class,
            "confidence": float(conf.item()),
            "similar_image_path": similar_img_path
        }

if __name__ == "__main__":
    train_model()
    
    create_embeddings_database()
    
    predictor = CavyPredictor()
    result = predictor.predict("testImg.jpg")
    print(result)
    pass