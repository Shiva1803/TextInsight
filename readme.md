TextInsight: Transformer-Based Sentiment and Thematic Analysis System

TextInsight is a transformer-driven NLP system that performs sentiment classification, thematic zero-shot categorization, text summarization, explainability, and dataset-level analytics. The application integrates multiple pre-trained models into a unified pipeline and provides an interactive Streamlit dashboard for analysis.

Overview

The system is designed to analyze user-generated text such as reviews, feedback, and short documents. It uses modern transformer architectures to:

• Classify sentiment
• Identify themes using zero-shot inference
• Generate summaries
• Provide token-level explanations
• Support batch-level analytics and visualization

Both single-text and CSV-based analysis workflows are supported.

Key Features
1. Sentiment Analysis

Uses the DistilBERT SST-2 model (distilbert-base-uncased-finetuned-sst-2-english) to classify text as Positive or Negative.

Outputs include:
• Sentiment label
• Confidence score
• Token-level explanation using attention-based scoring

2. Thematic Classification (Zero-Shot)

Uses BART-large-MNLI (facebook/bart-large-mnli) for theme prediction without any fine-tuning.

Supports customizable domains such as:
• Education
• Technology
• Healthcare
• Finance
• Entertainment
• AI
• User Experience

Outputs include the top theme, theme confidence, and a score breakdown for all candidate themes.

3. Text Summarization

Uses DistilBART CNN (sshleifer/distilbart-cnn-12-6) to summarize:
• Entire datasets
• Theme-specific text groups

4. Token-Level Explainability

Computes token contributions using attention weights and sentiment direction.
This helps users understand why the model selected a particular sentiment prediction.

5. Batch CSV Analysis

Users can upload a CSV containing a “text” column.
For each entry, the system:
• Predicts sentiment
• Assigns themes
• Computes scores
• Generates summaries
• Produces WordClouds and visual statistics

6. Visualizations

Includes:
• Sentiment distribution (Pie chart)
• Theme distribution (Bar chart)
• WordClouds for each theme

7. Model Evaluation

Includes utilities to compute:
• Accuracy
• Precision
• Recall
• F1-score
• Confusion matrix
• Classification report

Useful for validating system performance on labeled datasets.

Models Used
Sentiment Model

distilbert-base-uncased-finetuned-sst-2-english
A distilled BERT model fine-tuned on SST-2.

Theme Model (Zero-Shot)

facebook/bart-large-mnli
Based on BART trained on MNLI, ideal for natural language inference and zero-shot tasks.

Summarization Model

sshleifer/distilbart-cnn-12-6
A compressed BART model trained on CNN/DailyMail.

Tech Stack

Python
Streamlit
Hugging Face Transformers
PyTorch
Plotly Express
Matplotlib
WordCloud
Pandas, NumPy
scikit-learn

Project Structure

TextInsight/
├── app.py
├── model_pipeline.py
├── model_evaluation.py
├── evaluation_data.csv
├── requirements.txt
└── README.md

Running the Application

Install dependencies:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py

Running Evaluation

Prepare a CSV with:

text

true_sentiment

true_theme

Run:
python3 model_evaluation.py

This outputs accuracy, precision, recall, F1-score, and confusion matrices.