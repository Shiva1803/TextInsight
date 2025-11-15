Sentiment and Thematic Analysis System

This project is a multi-task NLP system that performs sentiment analysis, thematic (zero-shot) classification, text summarization, token-level explainability, and dataset-level analytics. It provides an end-to-end workflow from text preprocessing to visual insights and is deployed through a Streamlit-based interactive dashboard.

Overview

This system analyzes user-generated text such as reviews, feedback, and short documents. It uses transformer-based models to determine sentiment, identify underlying themes, generate summaries, and explain predictions at a token level.

The application supports:

Single text analysis
Batch analysis using CSV files
Token-level contribution visualization
Theme-wise summaries
WordCloud generation
Pie and bar chart visualizations
Evaluation using labeled datasets

This project combines multiple transformer models into a single unified analytics tool.

Features
1. Sentiment Analysis

Uses the DistilBERT SST-2 fine-tuned model to classify text as positive or negative.
Output includes:
    a. Sentiment label
    b. Confidence score
    c. Token-level explanation using attention-based scoring

2. Thematic Classification (Zero-Shot)

Uses a DistilBART MNLI model to infer themes without any training.
Supported themes include domains like:
education, technology, healthcare, finance, entertainment, AI, user experience

Outputs include:
    a. Top theme label
    b. Theme confidence score
    c. Score breakdown for all candidate themes

3. Text Summarization

Uses DistilBART CNN model to generate concise summaries of:
An entire dataset
Individual theme groups
Summaries condense the overall meaning while preserving key insights.

4. Token-Level Explainability

The system computes token contributions using attention weights and sentiment direction, enabling explainable NLP. This allows users to understand why the model chose a particular sentiment.

5. Batch Analysis (CSV Processing)

Users can upload a CSV file containing a text column. For each row, the system:
    a. Computes sentiment
    b.Predicts themes
    c. Aggregates statistics
    d. Generates summaries
    e. Creates WordCloud visualizations

6. Visualizations

Using Plotly and Matplotlib, the system displays:
    Sentiment distribution
    Theme frequency distribution
    WordClouds for each theme
These help present insights in an interpretable format.

7. Model Evaluation

The project includes evaluation functions to calculate:

    Accuracy
    Precision
    Recall
    F1-score
    Confusion matrix
    Classification report

This supports formal assessment using labeled datasets.

Models Used
Sentiment Model

distilbert-base-uncased-finetuned-sst-2-english
A distilled version of BERT fine-tuned on the SST-2 dataset.
Provides efficient and accurate sentiment classification.

Theme Model (Zero-Shot)

typeform/distilbert-base-uncased-mnli
Distilled from a BART MNLI model for natural language inference.
Enables theme prediction without labeled training data.

Summarization Model

sshleifer/distilbart-cnn-12-6
A compressed version of BART fine-tuned on CNN/DailyMail.
Used for abstractive summarization.

Tech Stack

    Python
    Streamlit for the interactive UI
    Hugging Face Transformers
    PyTorch
    Plotly Express
    Matplotlib
    WordCloud
    Pandas and NumPy
    scikit-learn (evaluation metrics)

File Structure
project/
│
├── app.py                     # Streamlit web application
├── model_pipeline.py          # Core model logic and evaluation utilities
├── evaluation_data.csv        # Optional test dataset
├── model_evaluation.py        # Script to compute evaluation metrics
├── requirements.txt           # Dependencies for deployment
└── README.md                  # Project documentation

Running the Application

    Install requirements:

    pip install -r requirements.txt


Run the Streamlit application:

    streamlit run app.py

    Running Evaluation

Prepare a CSV containing:

    1. text
    2. true_sentiment
    3. true_theme

Run:
python3 model_evaluation.py


The script prints accuracy, precision, recall, F1-score, and confusion matrices.

Use Cases
    This system is suitable for:
    Customer feedback analysis
    Product review mining
    Education technology analytics
    Market research
    Sentiment-driven insights
    Zero-shot thematic tagging
    Explainable AI demonstrations