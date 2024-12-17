import streamlit as st
from clearml import Task, TaskTypes  # Импортируем ClearML
import dashboard.utils.streamlit_utils as su


def main():

    # Инициализация ClearML Task
    task = Task.init(
        project_name="ML Service Dashboard",
        task_name="Streamlit User Session",
        task_type=TaskTypes.custom
    )
    task.logger.info("Streamlit Dashboard session started.")

    # User Authentication
    if not su.authenticate():
        return

    # Title
    st.title("HSE MLOps 2024 - ML Service")
    st.write("""Authors: *Брыш Павел, Хутиев Рустэм, Куликовских Денис*""")

    # Sidebar
    st.sidebar.write("""---""")
    su.log_off()

    st.sidebar.write("""---""")
    su.check_server_status()

    st.sidebar.write("""---""")
    st.sidebar.subheader("What Do You Want to Do?")
    section = st.sidebar.selectbox(
        "Please select the desired action:",
        ["Manage Datasets", "Create & Train Models", "Predict & Test Models", "Delete Models"],
    )
    st.sidebar.write("""---""")

    # Models View (always shown)
    st.write("""---""")
    st.subheader("Trained Models")

    if st.button("Refresh", key="refresh_models"):
        pass

    selected_models = su.view_models("multiple")
    su.view_models_hyperparams(selected_models=selected_models)
    st.write("""---""")

    # Action Sections
    if section == "Manage Datasets":
        st.subheader("Manage Datasets")
        st.markdown("""##### Storage""")
        datasets = su.view_datasets(selection_mode="multiple")
        st.markdown("""##### Load & Save""")
        su.load_and_save_dataset()
        st.markdown("""##### Delete""")
        su.delete_datasets(datasets)

    elif section == "Create & Train Models":
        st.subheader("Create & Train Models")
        task.logger.info("User initiated model creation and training pipeline.")
        su.create_and_train_pipeline()
        task.logger.info("Model creation and training completed.")

    elif section == "Predict & Test Models":
        st.subheader("Predict & Test Models")
        task.logger.info("User started prediction pipeline.")
        su.predict_and_test_pipeline(selected_models=selected_models)
        task.logger.info("Prediction pipeline completed.")

    elif section == "Delete Models":
        st.subheader("Delete Models")
        task.logger.info("User started model deletion pipeline.")
        su.delete_model(selected_models=selected_models)
        task.logger.info("Model deletion completed.")

    # Завершение задачи ClearML
    task.close()


# Firing up the dashboard
if __name__ == "__main__":
    main()
