from concurrent import futures
import grpc
import uuid
import service_pb2
import service_pb2_grpc
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging
from clearml import Task


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpc_server")

# Хранилище для моделей
models_store = {}

class ModelService(service_pb2_grpc.ModelServiceServicer):
    def TrainModel(self, request, context):

        # Инициализация задачи ClearML
        task = Task.init(project_name="Мой проект", task_name="gRPC TrainModel")
        task.connect(request.hyperparameters)


        if request.model_type == "linear_regression":
            model = LinearRegression()
        elif request.model_type == "random_forest":
            n_estimators = int(request.hyperparameters.get("n_estimators", 100))
            max_depth = int(request.hyperparameters.get("max_depth", None)) if request.hyperparameters.get("max_depth") else None
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Unsupported model type")
            task.close()
            return service_pb2.TrainResponse()

        # Обучение модели
        X = [row.features for row in request.training_data]
        y = request.target
        model.fit(X, y)

        # Логирование метрик
        task.set_metric("train_score", model.score(X, y))

        model_id = str(uuid.uuid4())
        models_store[model_id] = model

        task.close()
        return service_pb2.TrainResponse(model_id=model_id)

    def Predict(self, request, context):
        model = models_store.get(request.model_id)
        if model is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Model not found")
            return service_pb2.PredictResponse()

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
