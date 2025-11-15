# TextInsight: Transformer-Based Sentiment and Thematic Analysis System

TextInsight is a transformer-driven NLP system that performs sentiment classification, thematic zero-shot categorization, summarization, token-level explainability, and dataset-level analytics. The project integrates multiple pre-trained transformer models into a unified pipeline and provides an interactive Streamlit dashboard for real-time analysis.

## Overview

The system analyzes user-generated text such as reviews, feedback, comments, and short documents. Using modern transformer models, it can:

- Classify sentiment  
- Identify themes using zero-shot inference  
- Generate concise summaries  
- Provide token-level explanations  
- Support batch-level analytics and visualizations  

Both single-text and CSV-based workflows are supported.

## Key Features

### 1. Sentiment Analysis  
Model: **distilbert-base-uncased-finetuned-sst-2-english**  
Outputs:
- Sentiment label (Positive/Negative)  
- Confidence score  
- Token-level explanation using attention-based scoring  

### 2. Thematic Classification (Zero-Shot)

Model: **facebook/bart-large-mnli**  
Predicts themes without additional training.

Supported domains:
- Education  
- Technology  
- Healthcare  
- Finance  
- Entertainment  
- AI  
- User Experience  

Outputs:
- Top theme label  
- Theme confidence score  
- Score breakdown for each candidate theme  

### 3. Text Summarization

Model: **sshleifer/distilbart-cnn-12-6**  
Generates summaries for:
- Entire datasets  
- Theme-specific subsets  

### 4. Token-Level Explainability

Uses transformer attention weights to compute token contributions toward sentiment.

### 5. Batch CSV Analysis

For each entry, the system:
- Predicts sentiment  
- Assigns a theme  
- Computes scores  
- Generates summaries  
- Produces WordClouds and analytics  

### 6. Visualizations

Includes:
- Sentiment distribution (Pie chart)  
- Theme distribution (Bar chart)  
- WordClouds per theme  

### 7. Model Evaluation

Provides:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  
- Classification report  

## Models Used

### Sentiment Model
**distilbert-base-uncased-finetuned-sst-2-english**  
Distilled BERT fine-tuned on SST-2.

### Thematic Model (Zero-Shot)
**facebook/bart-large-mnli**  
BART model trained on MNLI for zero-shot classification.

### Summarization Model
**sshleifer/distilbart-cnn-12-6**  
Distilled BART trained on CNN/DailyMail.

## Tech Stack

- Python  
- Streamlit  
- Hugging Face Transformers  
- PyTorch  
- Plotly Express  
- Matplotlib  
- WordCloud  
- Pandas & NumPy  
- scikit-learn  

## Project Structure

```
TextInsight/
├── app.py
├── model_pipeline.py
├── model_evaluation.py
├── evaluation_data.csv
├── requirements.txt
└── README.md
```

## Running the Application

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

## Running Evaluation

Prepare a CSV with:

- text  
- true_sentiment  
- true_theme  

Then run:

```
python3 model_evaluation.py
```

This prints accuracy, precision, recall, F1-score, and confusion matrices.
