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

    selected_models = su.view_models()

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
        datasets = su.view_datasets(selection="multiple")

        st.markdown("""##### Load & Save""")
        su.load_and_save_dataset()

        st.markdown("""##### Delete""")
        su.delete_datasets(datasets)
    
    elif section == "Create & Train Models":
        
        st.subheader("Create & Train Models")
        su.create_and_train_pipeline()


        # if load_option == "Upload new CSV":
        #     st.write("""**Upload a new training set...**""")
        #     train_df = su.load_csv()
        #     if train_df is not None:
        #         features, targets, train_check = su.check_train_df(train_df)
        #     else:
        #         train_check = False
        #         features, targets = None, None
        # else:
        #     st.write("""**Select an existing dataset...**""")
        #     selected_datasets = su.view_datasets()

        #     if selected_datasets:
        #         dataset_name = selected_datasets[0]["dataset_name"]
        #         features, targets, train_check = su.get_formatted_dataset(dataset_name)
        #     else:
        #         train_check = False
        #         features, targets = None, None

        # if train_check:
        #     st.write("""**Now let's define the model...**""")
        #     su.create_and_train(features=features, targets=targets)

    elif section == "Predict & Test Models":
        st.subheader("Predict & Test Models")

        try:
            if len(selected_models) == 0:
                models_check = False
                st.warning("You have to train a model first.")
            else:
                models_check = True
        except TypeError:
            models_check = True

        if models_check:
            # Dataset loading options for prediction
            load_option = st.radio(
                "Choose how to load your test/prediction data:",
                ["Upload new CSV", "Use existing dataset"],
            )

            if load_option == "Upload new CSV":
                st.write("""**Upload a new test/prediction set...**""")
                predict_df = su.load_csv()
            else:
                st.write("""**Select an existing dataset...**""")
                selected_datasets = su.view_datasets()

                if selected_datasets:
                    dataset_name = selected_datasets[0]["dataset_name"]
                    predict_df = su.load_dataset_from_minio(dataset_name)
                else:
                    predict_df = None

            if predict_df is not None:
                target = st.toggle("My data has Target", value=True)
                features, targets, predict_test_check = su.check_predict_test_df(
                    df=predict_df, target=target
                )

                if predict_test_check:
                    st.write("""**Now let's make some predictions...**""")
                    predictions = su.predict(
                        selected_models=selected_models,
                        features=features,
                        df=predict_df,
                    )

                    st.write(
                        """**Finally, let's test the quality of trained model(s)...**"""
                    )
                    if target:
                        su.test(targets=targets, predictions=predictions)
                    else:
                        st.warning("You have to have a Target to test your model(s).")

    elif section == "Delete Models":
        st.subheader("Delete Models")

        su.delete_model(selected_models=selected_models)



# Firing up the dashboard
if __name__ == "__main__":
    main()
