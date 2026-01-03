import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(row):
    url = row['image_url']
    obs_id = row['id']
    species = row['scientific_name'] 
    
    if pd.isna(species) or pd.isna(url):
        return

    species_folder = species.replace(" ", "_")
    save_dir = os.path.join("dataset", species_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = os.path.join(save_dir, f"{obs_id}.jpg")
    
    if os.path.exists(filename):
        return

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        pass

def main_download():
    df = pd.read_csv("data.csv")
    print(f"Всего записей: {len(df)}")
    
    df = df.dropna(subset=['image_url', 'scientific_name'])
    
    rows = df.to_dict('records')
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        list(tqdm(executor.map(download_image, rows), total=len(rows)))

if __name__ == "__main__":
    main_download()