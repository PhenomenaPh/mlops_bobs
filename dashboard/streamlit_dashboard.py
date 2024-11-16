import streamlit as st
import dashboard.utils.func as db



def main():


    # User Authentification
    if not db.authenticate():
        return


    # Title
    st.title("HSE MLOps 2024 - ML Service")
    st.write("""Authors: *Брыш Павел, Хутиев Рустэм, Куликовских Денис*""")


    # Sidebar
    st.sidebar.write("""---""")

    db.log_off()

    st.sidebar.write("""---""")

    db.check_server_status()

    st.sidebar.write("""---""")

    st.sidebar.subheader("What Do You Want to Do?")
    section = st.sidebar.selectbox(
        "Please select the desired action:",
        ["Create & Train", "Predict & Test", "Delete"]
    )

    st.sidebar.write("""---""")
    

    # Models View (always shown)
    st.write("""---""")

    st.subheader("Trained Models")

    if st.button("Refresh"):
        pass

    selected_models = db.view_models()

    db.view_models_hyperparams(selected_models=selected_models)

    st.write("""---""")


    # Action Sections
    if section == "Create & Train":

        st.subheader("Create and Train")
        st.write("""**First, let's download the training set...**""")

        train_df = db.load_csv()

        features, targets, train_check = db.check_train_df(train_df)

        if train_check:
            st.write("""**Now let's define the model...**""")
            db.create_and_train(features=features, targets=targets)

    elif section == "Predict & Test":

        st.subheader("Predict and Test")

        try:
            if len(selected_models) == 0:
                models_check = False
                st.warning("You have to train a model first.")
            else:
                models_check = True
        except TypeError:
            models_check = True
        
        if models_check:

            st.write("""**First, let's download the predict/test set...**""")

            predict_df = db.load_csv()

            target = st.toggle("My data has Target", value=True)

            features, targets, predict_test_check = db.check_predict_test_df(df=predict_df, target=target)

            if predict_test_check:
                
                st.write("""**Now let's make some predictions...**""")
                predictions = db.predict(selected_models=selected_models, features=features, df=predict_df)

                st.write("""**Finally, let's test the quality of trained model(s)...**""")
                if target:
                    db.test(targets=targets, predictions=predictions)
                else:
                    st.warning("You have to have a Target to test your model(s).")

    elif section == 'Delete':

        st.subheader("Delete")
        
        db.delete_model(selected_models=selected_models)



# Firing up the dashboard
if __name__ == "__main__":
    main()
