import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Analysis Dashboard")

API_URL = "http://127.0.0.1:8000"

menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard","Analyze Tweet","Dataset Analytics"]
)

# Dashboard
if menu == "Dashboard":

    st.subheader("Project Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Tweets", "1.6M")
    col2.metric("Positive", "800K")
    col3.metric("Negative", "700K")
    col4.metric("Neutral", "100K")

    chart_data = pd.DataFrame({
        "Sentiment":["Positive","Negative","Neutral"],
        "Count":[800000,700000,100000]
    })

    fig = px.pie(
        chart_data,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# Analyze Tweet
elif menu == "Analyze Tweet":

    tweet = st.text_area("Enter Tweet")

    if st.button("Analyze"):

        response = requests.post(
            f"{API_URL}/analyze/text",
            json={"text":tweet}
        )

        data = response.json()

        st.success(
            f"Sentiment : {data['sentiment']}"
        )

        st.write("Confidence Score :", data["score"])

# Dataset Analytics
elif menu == "Dataset Analytics":

    st.subheader("Dataset Analytics")

    df = pd.read_csv(
        "training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        header=None
    )

    st.write("Dataset Shape :", df.shape)

    st.dataframe(df.head())

    sentiment_counts = {
        "Positive": len(df[df[0]==4]),
        "Negative": len(df[df[0]==0])
    }

    fig = px.bar(
        x=list(sentiment_counts.keys()),
        y=list(sentiment_counts.values()),
        title="Dataset Sentiment Count"
    )

    st.plotly_chart(fig, use_container_width=True)