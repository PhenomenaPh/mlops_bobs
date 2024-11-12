import requests

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt

from requests.auth import HTTPBasicAuth
from st_aggrid import AgGrid, GridOptionsBuilder
from time import sleep



# Пользовательские данные для авторизации
USERNAME = "hse_mlops_2024"
PASSWORD = "strong_password"
# URL API сервиса
API_BASE_URL = "http://0.0.0.0:8000/api/v1"



def authenticate():
    """
    Функция для авторизации пользователя

    - Производит аутентификацию пользователя
    - Пропускает авторизацию, если за последние 15 минут был успешный вход
    """
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "last_login" not in st.session_state:
        st.session_state.last_login = dt.datetime.now()
    if (dt.datetime.now() - st.session_state.last_login).total_seconds() > 60 * 15:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        login_clicked = st.button("Login")
        if login_clicked and username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.session_state.last_login = dt.datetime.now()
            st.rerun()
        
        elif login_clicked:
            st.error("Invalid credentials. Please try again.")
        
        return False
    
    st.session_state.last_login = dt.datetime.now()

    return True


def log_off():
    """
    Функция для выхода пользователя из своей учетной записи

    - Выходит из учетной записи пользователя после нажатия соответствующей кнопки в боковом меню
    """

    st.sidebar.subheader("Your Profile")
    st.sidebar.write(f"""Hello, ***{USERNAME}*** !""")

    logoff_clicked = st.sidebar.button("Log Off")
    if logoff_clicked:
        
        st.session_state.authenticated = False
        st.rerun()


def check_server_status():
    """
    Функция для проверки статуса API сервера

    - Проверяет статус сервера после нажатия соответствующей кнопки в боковом меню
    - Выводит результат проверки в боковое меню
    - Выводит дату и время последней проверки в боковое меню
    """
    
    st.sidebar.subheader("Check Server Status")
    check_status_clicked = st.sidebar.button("Check")

    if check_status_clicked:

        try:
            response = requests.get(f"{API_BASE_URL}/health", auth=HTTPBasicAuth(USERNAME, PASSWORD))
            
            if response.status_code == 200:
                st.sidebar.success("Server is up and running.")
            
            else:
                st.sidebar.error(f"Server is not available. Status code: {response.status_code}.")
        
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error connecting to the server: {e}")

        last_check_dttm = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.write(f"""*Last checked: {last_check_dttm}*""")


def get_models() -> pd.DataFrame:
    """
    Функция для загрузки обученных моделей с сервера

    - Дергает ручку API сервера, которая выгружает обученные модели
    - Если обученных моделей нет, выводит предупреждение
    - Если модели есть, преобразует полученные данные в pd.DataFrame и отдает его
    - Если сервер выдает ошибку, выводит соответствующее сообщение
    """

    response = requests.get(f"{API_BASE_URL}/models", auth=HTTPBasicAuth(USERNAME, PASSWORD))
    
    if response.status_code == 200:
        
        models = response.json()
        
        if not models:
            st.warning("No models available. Let's train your first one!")
            return
        
        models_df = pd.DataFrame(models)
        models_df.created_at = (pd.to_datetime(models_df.created_at) + dt.timedelta(hours=3)).dt.strftime('(MSK) %Y-%m-%d %H:%M:%S')
        models_df.columns = [
            "Model ID",
            "Model Name",
            "Model Type",
            "Created At",
            "Hyperparameters"
        ]

        return models_df
    
    else:
        st.error(f"Error fetching models. Response: {response.status_code}.")


def view_models() -> list:
    """
    Функция для просмотра и выбора обученных моделей

    - Выгружает обученные модели с сервера
    - Формирует интерактивную таблицу просмотра метаданных обученных моделей
    - Возвращает метаданные выбранных клиентом моделей
    """

    models_df = get_models()

    if models_df is None:

        return []
    
    else:

        gb = GridOptionsBuilder.from_dataframe(models_df[["Model Name", "Model Type", "Created At", "Model ID"]])
        gb.configure_selection("multiple", use_checkbox=True)
        grid_options = gb.build()

        response = AgGrid(
            models_df,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode="MODEL_CHANGED",
            height=200,
            theme="streamlit"
        )

        return response["selected_rows"]


def view_models_hyperparams(selected_models: list):
    """
    Функция для просмотра гиперпараметров выбранных обученных моделей

    - Принимает выбранные клиентом модели
    - Для этих моделей выводит гиперпараметры по группам моделей
    """
    
    if len(selected_models) == 0:
        return
    else:
        models_df = get_models()

    if models_df.shape[0] > 0 and selected_models is None:
        st.write("""*Select models in the table above to view their hyperparameters*""")
    
    elif models_df.shape[0] > 0:
        
        lin_reg = selected_models[selected_models['Model Type'] == 'LinearRegression']
        forest = selected_models[selected_models['Model Type'] == 'RandomForest']
            
        if lin_reg.shape[0] > 0:
                
            st.write("LinearRegression")
            lin_reg_hp_df = pd.concat(
                [
                    lin_reg['Model Name'],
                    lin_reg['Hyperparameters'].apply(lambda x: pd.Series(x))
                ],
                axis=1
            )
            st.dataframe(lin_reg_hp_df, hide_index=True)
            
        if forest.shape[0] > 0:
                
            st.write("RandomForest")
            forest_hp_df = pd.concat(
                [
                    forest['Model Name'],
                    forest['Hyperparameters'].apply(lambda x: pd.Series(x))
                ],
                axis=1
            )
            st.dataframe(forest_hp_df, hide_index=True)


def load_csv() -> pd.DataFrame:
    """
    Функция для загрузки данных в .csv формате

    - Загружает предоставленный клиентом .csv файл
    - Считывает его в pd.DataFrame
    - Переводит датафрейм в np.float64, тк того требует спецификация API сервиса
    - Проверяет, что датафрейм не пустой
    - При неудачной проверке выводит сообщение об ошибке
    """

    csv_file = st.file_uploader(f"""Choose a `.csv` file to upload:""", type="csv")

    if csv_file is not None:

        try:
            df = pd.read_csv(csv_file)
            
            try:
                df = df.astype(np.float64)
            except Exception as e:
                st.error(f"Dataframe has to consist of floats: {e}")
                return
            
            if df.shape[0] == 0:
                st.error("Dataframe is empty.")
                return
            
            st.success("Dataframe loaded successfully.")
            
            return df

        except Exception as e:
            st.error(f"""Error reading `.csv` file: {e}""")


def check_train_df(df: pd.DataFrame):
    """
    Функция для проверки загруженного train датасета и его форматирования

    - Проверяет, что в датасете есть хотя бы одна фича и таргет
    - Берет последний столбец датасета в качестве таргета
    - Выводит справочную информацию о датасете, чтобы клиент убедился, что данные считаны верно
    - Возвращает отформатированные фичи, таргет и булево значение, отражающее результат проверки
    """

    if df is None:
        return None, None, False
    
    elif df.shape[1] < 2:
        st.error(f"Your dataframe has only 1 column. Must include at least 1 Feature and a Target.")
        return None, None, False
    
    else:

        st.warning("Please note that the last column will be treated as Target.")
        
        st.write(f"Train set shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write(f"""- Target: `{df.columns[-1]}`\n- {df.shape[1]-1} Feature(s): `{', '.join(df.columns[:-1])}`""")
        st.write("Your train set:")
        st.dataframe(df, height=210)

        return df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist(), True


def create_model(data: dict) -> str:
    """
    Функция для создания модели

    - Дергает ручку API сервера, которая создает модель
    """

    response = requests.post(f"{API_BASE_URL}/models", json=data, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    
    if response.status_code == 201:
        st.success(f"""Model ***{data["model_name"]}*** successfully created.""")
        models_df = get_models()
        return models_df.set_index("Model Name").loc[data["model_name"], "Model ID"]
    
    else:
        st.error(f"Error creating model. Response: {response.status_code}.")
        return None


def train_model(model_id: str, model_name: str, data: dict):
    """
    Функция для обучения модели

    - Дергает ручку API сервера, которая обучает модель
    """

    response = requests.post(f"{API_BASE_URL}/models/{model_id}/train", json=data, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    
    if response.status_code == 200:
        sleep(1)
        st.success(f"""Model ***{model_name}*** successfully trained.""")
        st.rerun()
    
    else:
        st.error(f"Error training model. Response: {response.status_code}.")


def create_and_train(features: list, targets: list):
    """
    Функция для создания и обучения модели

    - Собирает от клиента спецификацию модели
    - Дергает ручку API сервера, которая создает модель
    - Дергает ручку API сервера, которая обучает модель
    """

    models_df = get_models()

    model_type = st.selectbox("What model will we train?", ["LinearRegression", "RandomForest"])
    
    model_name = st.text_input("How would you name it?")
    if (models_df is not None) and (model_name in models_df["Model Name"].values.tolist()):
        st.warning(f"""Model **{model_name}** already exists. Training will override it.""")
    
    st.write("""**Finally, let's play with hyperparameters...**""")

    if model_type == "LinearRegression":

        fit_intercept = st.toggle(
            """Fit Intercept | *Whether to calculate the intercept for this model.*""",
            value=True
        )
        copy_X = st.toggle(
            """Copy X | *If `True`, X will be copied; else, it may be overwritten.*""",
            value=True
        )
        positive = st.toggle(
            """Positive Coefficients | *When set to `True`, forces the coefficients to be positive.*""",
            value=False
        )
        n_jobs = st.toggle(
            """CPU Cores | *The number of jobs to use for the computation.*""",
            value=False
        )
        if n_jobs:
            n_jobs = st.number_input(
                """CPU Cores""",
                min_value=1, max_value=4, step=1, value=1
            )
        else:
            n_jobs = None
        
        hyperparameters = {
            "fit_intercept": fit_intercept,
            "copy_X": copy_X,
            "n_jobs": n_jobs,
            "positive": positive
        }
        
    elif model_type == "RandomForest":
        
        n_estimators = st.slider(
            """Number of Estimators | *The number of trees in the forest.*""",
            min_value=100, max_value=500, step=10, value=100
        )
        max_depth = st.toggle(
            """Max Depth | *The maximum depth of the tree.*""",
            value=False
        )
        if max_depth:
            max_depth = st.slider(
                """Max Depth""",
                min_value=100, max_value=1000, step=50, value=100
            )
        else:
            max_depth = None
        min_samples_split = st.slider(
            """Min Samples Split | *The minimum number of samples required to split an internal node.*""",
            min_value=2, max_value=10, step=1, value=2
        )
        min_samples_leaf = st.slider(
            """Min Samples Leaf | *The minimum number of samples required to be at a leaf node.*""",
            min_value=1, max_value=5, step=1, value=1
        )
        max_features = st.selectbox(
            """Max Features | *The number of features to consider when looking for the best split.*""",
            ["sqrt", 'log2']
        )
        random_state = st.number_input(
            """Random State | *For reproductive results.*""",
            min_value=1, max_value=1000000, step=1, value=42
        )
        n_jobs = st.toggle(
            """CPU Cores | *The number of jobs to use for the computation.*""",
            value=False
        )
        if n_jobs:
            n_jobs = st.number_input(
                """CPU Cores""",
                min_value=1, max_value=4, step=1, value=1
            )
        else:
            n_jobs = None
        
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs
        }

    if st.button("Train Model"):

        if model_name == "":
            st.error("Model name is required.")
            return
        
        create_data = {
            "model_name": model_name,
            "model_type": model_type,
            "hyperparameters": hyperparameters
        }
        model_id = create_model(data=create_data)

        with st.spinner(f"""Training ***{create_data["model_name"]}*** model..."""):
            if model_id is None:
                st.error(f"""Error getting ***{create_data["model_name"]}*** model after creation.""")
            else:
                train_data = {
                    "features": features,
                    "targets": targets
                }
                train_model(model_id=model_id, model_name=create_data["model_name"], data=train_data)


def delete_model(selected_models):
    """
    Функция для удаления выбранных моделей

    - Принимает выбранные клиентом модели
    - Для каждой из этих моделей дергает ручку API сервера, которая удаляет модель
    """

    if len(selected_models) == 0:
        st.warning("There are no models to delete yet.")
        return
    elif selected_models is None:
        st.write("""*Select models in the table above you want to delete*""")
        return
    else:
        st.write("Press the button below to delete selected model(s)")
    
    if "delete_model_pending" not in st.session_state:
        st.session_state.delete_model_pending = False
    
    if st.button("Delete") or st.session_state.delete_model_pending:
        
        st.session_state.delete_model_pending = True

        st.write(f"Following model(s): [{', '.join(selected_models['Model Name'])}] will be deleted forever")
        if st.button("Confirm Delete"):
            
            for model_id in selected_models["Model ID"].values.tolist():
                
                response = requests.delete(f"{API_BASE_URL}/models/{model_id}", auth=HTTPBasicAuth(USERNAME, PASSWORD))
                
                if response.status_code == 200:
                    st.success(f"""Model ***{model_id}*** deleted successfully.""")
                
                else:
                    st.error(f"""Error deleting model ***{model_id}***. Response: {response.status_code}.""")
            
            st.session_state.delete_model_pending = False
            st.rerun()
