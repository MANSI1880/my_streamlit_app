import pandas as pd
import pickle


# Load model
def load_model():
    with open("model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["columns"]


# Preprocess user input
def preprocess_input(user_dict, columns):
    user_df = pd.DataFrame([user_dict])
    user_df = pd.get_dummies(user_df, drop_first=True)
    user_df = user_df.reindex(columns=columns, fill_value=0)
    return user_df


# Predict
def make_prediction(user_dict):
    model, columns = load_model()
    final_df = preprocess_input(user_dict, columns)
    pred = model.predict(final_df)[0]
    return pred
