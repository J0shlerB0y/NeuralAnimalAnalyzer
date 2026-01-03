# Neural Caviidae Recognizer

Веб-приложение для классификации животных семейства Свинковые

---

## Инструкция по запуску

### Шаг 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/NeuralAnimalAnalyzer.git
cd NeuralAnimalAnalyzer
```

### Шаг 2. Подготовка данных и обучение модели

Перейдите в директорию нейросети и установите зависимости:

```bash
cd NeuralNetwork
pip install pandas requests tqdm torch torchvision scikit-learn pillow numpy
```

1.  **Загрузка датасета:**
    Запустите скрипт загрузки изображений. Это скачает необходимые данные в папку `dataset`.
    ```bash
    python downloadImg.py
    ```

2.  **Обучение и индексация:**
    Запустите скрипт обучения. Он создаст файлы весов (`cavy_model.pth`), базу векторов (`database_embeddings.npy`) и маппинг классов.
    *Откройте файл `model_logic.py`, раскомментируйте вызов `run_training_pipeline()` в конце файла и запустите:*
    ```bash
    python model_logic.py
    ```
    *После завершения убедитесь, что в папке появились файлы `.pth`, `.npy`, `.json`.*

### Шаг 3. Запуск в Docker

Вернитесь в корневую директорию проекта (где находится `docker-compose.yml`) и выполните сборку контейнеров:

```bash
cd ..
docker-compose up --build
```

### Шаг 4. Использование

После успешного запуска контейнеров веб-интерфейс доступен по адресу:

**[http://localhost](http://localhost)** (порт 80)
