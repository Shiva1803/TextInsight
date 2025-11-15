# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from model_pipeline import load_models, analyze_single_text, analyze_batch, get_token_sentiment_scores, generate_summary

st.set_page_config(
    page_title="Sentiment & Thematic Analyzer",
    layout="centered"
)

st.title("Sentiment and Thematic Analysis")
st.caption("via DistilBERT (Sentiment) + BART Zero-Shot (Theme)")


with st.spinner("Loading models..."):
    sentiment_pipe, theme_pipe, summarizer, tokenizer, model = load_models()


st.sidebar.title("Settings")
candidate_labels = st.sidebar.multiselect(
    "Theme Categories",
    ["education", "technology", "healthcare", "finance",
     "entertainment", "AI", "user experience"],
    default=["education", "technology", "AI"]
)

st.sidebar.info("You can customize the theme labels here.")

st.subheader("Analyze a Single Sentence")

text = st.text_area("Enter your text", "The platform is user-friendly and effective.")

if st.button("Analyze Text"):
    if text.strip():
        with st.spinner("Analyzing..."):
            sent_label, sent_score, top_theme, top_theme_score, all_theme_scores = analyze_single_text(
                text, sentiment_pipe, theme_pipe, candidate_labels
            )
            
            # Get token-level sentiment scores
            token_scores, pred_class = get_token_sentiment_scores(text, tokenizer, model)

        st.success(f"**Sentiment:** {sent_label} (score: {sent_score:.2f})")
        st.progress(sent_score)
        
        st.info(f"**Top Theme:** {top_theme} (score: {top_theme_score:.2f})")
        st.progress(top_theme_score)
        
        # Theme score breakdown
        st.subheader("Theme Score Breakdown")
        theme_all_scores = {label: round(score, 3) for label, score in all_theme_scores.items()}
        st.json(theme_all_scores)
        
        # Raw model output toggle
        if st.checkbox("Show raw model output"):
            st.subheader("Raw Model JSON Output")
            raw_sentiment = sentiment_pipe(text)[0]
            raw_theme = theme_pipe(text, candidate_labels)
            
            st.write("**Sentiment Model Output:**")
            st.json(raw_sentiment)
            
            st.write("**Theme Model Output:**")
            st.json(raw_theme)
        
        # Display token-level explanation
        st.subheader("Token-Level Sentiment Explanation")
        st.caption("See which words contributed to the sentiment classification")
        
        # Create HTML for highlighted text
        if token_scores:
            html_parts = []
            max_score = max([abs(s) for _, s in token_scores])
            
            for token, score in token_scores:
                # Normalize score for color intensity
                abs_score = abs(score)
                intensity = min(abs_score / max_score, 1.0) if max_score > 0 else 0
                
                # Color based on sentiment contribution
                if score > 0:
                    # Positive contribution - green
                    color = f"rgba(0, 200, 0, {intensity * 0.6})"
                else:
                    # Negative contribution - red
                    color = f"rgba(255, 0, 0, {intensity * 0.6})"
                
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; '
                    f'border-radius: 3px; display: inline-block;">{token}</span>'
                )
            
            html_output = f'<div style="line-height: 2.5; font-size: 16px;">{"".join(html_parts)}</div>'
            st.markdown(html_output, unsafe_allow_html=True)
            
            st.markdown("""
                <span style="color: green; font-weight: bold;">Green = Positive Contribution</span>  
                <span style="color: red; font-weight: bold;">Red = Negative Contribution</span>
            """, unsafe_allow_html=True)
        else:
            st.warning("Could not generate token-level explanation for this text.")
    else:
        st.warning("Please enter some text to analyze.")

st.subheader("Batch Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV containing a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column!")
    else:
        if st.button("Run Batch Analysis"):
            with st.spinner("Processing entire dataset..."):
                df_results = analyze_batch(df, sentiment_pipe, theme_pipe, candidate_labels)

            st.success("Batch analysis complete!")
            
            # ---- Overall Summary ----
            st.write("### Overall Summary of Reviews")
            with st.spinner("Generating summary..."):
                overall_summary = generate_summary(df["text"].tolist(), summarizer)
            st.info(overall_summary)

            st.write("Results:")
            st.dataframe(df_results.head())

            # ---- Sentiment Pie Chart ----
            st.write("### Sentiment Distribution")
            sentiment_counts = df_results["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig = px.pie(sentiment_counts, values="count", names="sentiment", title="Sentiment Breakdown")
            st.plotly_chart(fig)

            # ---- Theme Distribution ----
            st.write("### Theme Distribution")
            theme_counts = df_results["theme"].value_counts().reset_index()
            theme_counts.columns = ["theme", "count"]
            fig2 = px.bar(theme_counts, x="theme", y="count", title="Theme Counts")
            st.plotly_chart(fig2)
            
            # ---- WordCloud Per Theme ----
            st.write("### WordCloud & Summary Per Theme")
            st.caption("Visualize dominant words and key insights for each theme category")
            
            unique_themes = df_results["theme"].unique()
            
            for theme in unique_themes:
                # Get all text for this theme
                theme_texts = df_results[df_results["theme"] == theme]["text"].tolist()
                combined_text = " ".join(theme_texts)
                
                if combined_text.strip():
                    st.subheader(f"Theme: {theme}")
                    
                    # Skip themes with very few entries
                    if len(theme_texts) < 3:
                        st.write("Not enough data to summarize.")
                        continue
                    
                    # Generate theme-specific summary
                    with st.spinner(f"Generating summary for {theme}..."):
                        theme_summary = generate_summary(theme_texts, summarizer, max_length=100, min_length=30)
                    st.markdown(f"**Summary:** {theme_summary}")
                    
                    # Generate WordCloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="viridis",
                        stopwords=set(STOPWORDS),
                        max_words=50
                    ).generate(combined_text)
                    
                    # Display using matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("---")

st.markdown("---")
st.caption("Built by Shivansh and Ishaan")
