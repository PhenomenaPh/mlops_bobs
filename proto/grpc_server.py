from concurrent import futures
import grpc
import uuid
from proto import service_pb2, service_pb2_grpc
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpc_server")

# Хранилище для моделей
models_store = {}

class ModelService(service_pb2_grpc.ModelServiceServicer):
    def TrainModel(self, request, context):
        if request.model_type == "linear_regression":
            model = LinearRegression()
        elif request.model_type == "random_forest":
            n_estimators = int(request.hyperparameters.get("n_estimators", 100))
            max_depth = int(request.hyperparameters.get("max_depth", None))
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Unsupported model type")
            return service_pb2.TrainResponse()

        # Обучение модели
        X = [row.features for row in request.training_data]
        y = request.target
        model.fit(X, y)

        model_id = str(uuid.uuid4())
        models_store[model_id] = model
        return service_pb2.TrainResponse(model_id=model_id)


    def Predict(self, request, context):
        # Проверка наличия модели
        model = models_store.get(request.model_id)
        if model is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return service_pb2.PredictResponse()

        # Преобразование данных и выполнение предсказания
        X = [row.features for row in request.data]
        predictions = model.predict(X)

        return service_pb2.PredictResponse(predictions=predictions.tolist())

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC сервер запущен на порту 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
