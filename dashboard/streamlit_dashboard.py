import streamlit as st

import dashboard.utils.streamlit_utils as su



def main():


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

        st.markdown("""
        ##### Contents
        - [Storage](#storage)
        - [Load & Save](#load-save)
        - [Delete](#delete)
        """)

        st.markdown("""##### Storage""")
        datasets = su.view_datasets(selection_mode="multiple")

        st.markdown("""##### Load & Save""")
        su.load_and_save_dataset()

        st.markdown("""##### Delete""")
        su.delete_datasets(datasets)
    
    elif section == "Create & Train Models":
        
        st.subheader("Create & Train Models")
        su.create_and_train_pipeline()

    elif section == "Predict & Test Models":
        
        st.subheader("Predict & Test Models")
        su.predict_and_test_pipeline(selected_models=selected_models)

    elif section == "Delete Models":
        
        st.subheader("Delete Models")
        su.delete_model(selected_models=selected_models)



# Firing up the dashboard
if __name__ == "__main__":
    main()
