import grpc
from concurrent import futures
import animal_pb2
import animal_pb2_grpc
from predictor import CavyPredictor

class AnimalRecognizerService(animal_pb2_grpc.AnimalRecognizerServicer):
    def __init__(self):
        self.predictor = CavyPredictor()

    def IdentifySpecies(self, request, context):
        if not request.image_data:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Image data is empty')
            return animal_pb2.RecognitionResponse()

        result = self.predictor.predict_bytes(request.image_data)

        if result:
            return animal_pb2.RecognitionResponse(
                species_name=result['species'],
                confidence=result['confidence'],
                similar_image_path=result['path'],
                similar_image_data=result['similar_bytes']
            )
        else:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Prediction failed')
            return animal_pb2.RecognitionResponse()

def serve():
    print("Запуск gRPC сервера...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    animal_pb2_grpc.add_AnimalRecognizerServicer_to_server(
        AnimalRecognizerService(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Сервер слушает на порту 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()