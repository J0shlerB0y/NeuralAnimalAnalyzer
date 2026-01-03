FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py predictor.py animal_pb2.py animal_pb2_grpc.py ./

COPY cavy_model.pth .
COPY class_mapping.json .
COPY database_embeddings.npy .
COPY database_paths.npy .

COPY dataset ./dataset

EXPOSE 50051

CMD ["python", "-u", "server.py"]