# import streamlit as st
# from textblob import TextBlob  # sentiment analysis package
# import plotly.graph_objects as go  # data visualization package

# st.markdown("# Welcome to my app!")

# st.markdown("### Please enter some text below and press enter!")

# st.markdown("### Pretty please!")


# def sent_app(text):
#     # Extract sentiment
#     sentiment_score = TextBlob(f"{text}").sentiment.polarity

#     # Use score to assign label
#     if sentiment_score > 0.15:
#         label = "Positive"
#     elif sentiment_score < -0.15:
#         label = "Negative"
#     else:
#         label = "Neutral"

#     # Plot score on gauge plot
#     fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=sentiment_score,
#             title={"text": f"Sentiment: {label}"},
#             gauge={
#                 "axis": {"range": [-1, 1]},
#                 "steps": [
#                     {"range": [-1, -0.15], "color": "red"},
#                     {"range": [-0.15, 0.15], "color": "gray"},
#                     {"range": [0.15, 1], "color": "lightgreen"},
#                 ],
#                 "bar": {"color": "yellow"},
#             },
#         )
#     )

#     return st.plotly_chart(fig)


# text = st.text_input("Enter text:", value="Enter text here")

# sent_app(text)


import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Goal
# The app should take user input for the features included in the model.
# The app should return (1) whether the person would be classified as a LinkedIn user or not and (2) the probability that the person uses LinkedIn.

# Notes
# How to take user input for the features included in the model?


# don't want text input for features (want proper labels)
# add probability value to classification


file_name = "social_media_usage.csv"

target_column = "sm_li"

features = ["income", "education", "parent", "married", "female", "age"]

codebook = {
    "income": [
        "Less than $10,000",
        "10 to under $20,000",
        "20 to under $30,000",
        "30 to under $40,000",
        "40 to under $50,000",
        "50 to under $75,000",
        "75 to under $100,000",
        "100 to under $150,000",
        "$150,000 or more",
    ],
    "education": [
        "Less than high school",
        "High school incomplete",
        "High school graduate",
        "Some college, no degree",
        "Two-year associate degree from a college or university",
        "Four-year college or university degree/bachelor's degree",
        "Some postgraduate or professional schooling, no postgraduate degree",
        "Postgraduate or professional degree, including master's, doctorate, medical or law degree",
    ],
    "parent": ["Yes", "No"],
    "married": [
        "Married",
        "Living with a partner",
        "Divorced",
        "Separated",
        "Widowed",
        "Never been married",
    ],
    "female": ["male", "female", "other"],
}


def get_model():
    # first dataframe
    s = pd.read_csv(file_name)

    # second dataframe
    ss = pd.DataFrame(
        {
            target_column: np.where(
                s["web1h"] == 1,
                1,
                0,
            ),
            "income": np.where(s["income"] > 9, np.nan, s["income"]),
            "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
            "parent": np.where(s["par"] == 1, 1, 0),
            "married": np.where(s["marital"] == 1, 1, 0),
            "female": np.where(s["gender"] == 2, 1, 0),
            "age": np.where(s["age"] > 98, np.nan, s["age"]),
        }
    )

    # drop missing values
    ss = ss.dropna(how="any", axis=0)

    # target vector
    y = ss[target_column]

    # feature set
    X = ss[features]

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=987
    )

    # logistic regression model
    lr = LogisticRegression(class_weight="balanced")

    # fit model with training data
    lr.fit(X_train, y_train)

    return lr


def main():
    st.markdown("# Are you a LinkedIn user?")

    st.markdown("### Modify this form to see my prediction!")

    model = get_model()

    income = st.selectbox("Income (household)", codebook["income"])
    education = st.selectbox(
        "Education (highest level completed)", codebook["education"]
    )
    parent = st.selectbox(
        "Parent (are you a parent of a child under 18 living in your home?)",
        codebook["parent"],
    )
    married = st.selectbox("Marital status", codebook["married"])
    female = st.selectbox("Gender", codebook["female"])
    age = st.slider("Age", 0, 98)

    person = {
        "income": [codebook["income"].index(income) + 1],
        "education": [codebook["education"].index(education) + 1],
        "parent": [1 if codebook["parent"].index(parent) + 1 == 1 else 0],
        "married": [1 if codebook["married"].index(married) + 1 == 1 else 0],
        "female": [1 if codebook["female"].index(female) + 1 == 2 else 0],
        "age": [age],
    }

    person_df = pd.DataFrame(person)

    is_linkedin_user = model.predict(person_df)

    prediction = (
        "😊 You are a LinkedIn user!"
        if is_linkedin_user == 1
        else "😔 You are not a LinkedIn user."
    )

    st.markdown(f"## {prediction}")


main()
