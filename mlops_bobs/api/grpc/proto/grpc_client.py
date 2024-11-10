import grpc
import os
import pandas as pd
from proto import service_pb2, service_pb2_grpc


def load_data(file_path):
    if not os.path.exists(file_path):
        print("Файл данных не найден, используется заглушка.")
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': [7.0, 8.0, 9.0],
            'target': [10.0, 11.0, 12.0]
        })
    else:
        data = pd.read_csv(file_path)

    target = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    training_data = [
        service_pb2.TrainingRow(features=row.tolist())
        for _, row in features.iterrows()
    ]

    return training_data, target.tolist()


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.ModelServiceStub(channel)

        training_data, target = load_data("data.csv")

        # Обучение модели с выбором типа и настройкой гиперпараметров
        train_request = service_pb2.TrainRequest(
            model_type="random_forest",
            hyperparameters={"n_estimators": "100", "max_depth": "5"},
            training_data=training_data,
            target=target  # Передаем целевую переменную
        )

        train_response = stub.TrainModel(train_request)
        print(f"Model trained with ID: {train_response.model_id}")

        # Предсказание на основе обученной модели
        predict_request = service_pb2.PredictRequest(
            model_id=train_response.model_id,
            data=training_data
        )
        predict_response = stub.Predict(predict_request)
        print(f"Predictions: {predict_response.predictions}")

if __name__ == '__main__':
    run()
