
import streamlit as st
from transformers import pipeline
import plotly.express as px


st.title("Sentiment + Thematic Analysis App")
st.write("Analyze text sentiment and infer its theme using DistilBERT & Zero-Shot Classification.")

@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    theme_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return sentiment_pipe, theme_pipe

sentiment_pipe, theme_pipe = load_models()


st.subheader("Analyze a Single Text")
text = st.text_area("Enter your text:", "The platform is user-friendly and effective.")

candidate_labels = ["education", "technology", "healthcare", "finance", "entertainment", "AI", "user experience"]

if st.button("Analyze"):
    with st.spinner("Running analysis..."):
        sent = sentiment_pipe(text)[0]
        theme = theme_pipe(text, candidate_labels)
        st.success(f"**Sentiment:** {sent['label']} (score: {sent['score']:.2f})")
        st.write(f"**Top Theme:** {theme['labels'][0]} (score: {theme['scores'][0]:.2f})")


st.subheader(" Batch Analysis (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    df["sentiment"] = df["text"].apply(lambda x: sentiment_pipe(x)[0]["label"])
    df["theme"] = df["text"].apply(lambda x: theme_pipe(x, candidate_labels)["labels"][0])

    st.write("Analysis Complete")
    st.dataframe(df.head())

    # Visualization
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    fig = px.pie(sentiment_counts, values="sentiment", names="index", title="Sentiment Distribution")
    st.plotly_chart(fig)
