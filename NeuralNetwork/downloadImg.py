import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm # для прогресс бара

def download_image(row):
    url = row['image_url']
    obs_id = row['id']
    # Используем научное название как имя класса (папки)
    species = row['scientific_name'] 
    
    # Если названия нет или ссылка битая — пропускаем
    if pd.isna(species) or pd.isna(url):
        return

    # Убираем пробелы и спецсимволы из названия папки
    species_folder = species.replace(" ", "_")
    save_dir = os.path.join("dataset", species_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, f"{obs_id}.jpg")
    
    if os.path.exists(filename):
        return # Уже скачано

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        pass # Игнорируем ошибки сети

def main_download():
    df = pd.read_csv("data.csv")
    print(f"Всего записей: {len(df)}")
    
    # Фильтруем, чтобы были url и имя вида
    df = df.dropna(subset=['image_url', 'scientific_name'])
    
    # Конвертируем dataframe в список словарей для итерации
    rows = df.to_dict('records')
    
    # Качаем в 20 потоков
    with ThreadPoolExecutor(max_workers=30) as executor:
        list(tqdm(executor.map(download_image, rows), total=len(rows)))

if __name__ == "__main__":
    main_download()