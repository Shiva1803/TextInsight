import model_pipeline
print("Loaded model_pipeline from:", model_pipeline.__file__)


import pandas as pd
from model_pipeline import load_models, evaluate_sentiment, evaluate_theme

sentiment_pipe, theme_pipe, summarizer, tokenizer, model = load_models()

df = pd.read_csv("evaluation_data.csv")


acc, prec, rec, f1, cm, report = evaluate_sentiment(df, sentiment_pipe)

print("SENTIMENT MODEL PERFORMANCE")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


candidate_labels = ["education", "technology", "healthcare", "finance",
                    "entertainment", "AI", "user experience"]

acc2, prec2, rec2, f12, cm2, report2 = evaluate_theme(df, theme_pipe, candidate_labels)

print("\n THEME MODEL PERFORMANCE")
print("Accuracy:", acc2)
print("Precision:", prec2)
print("Recall:", rec2)
print("F1 Score:", f12)
print("\nConfusion Matrix:\n", cm2)
print("\nClassification Report:\n", report2)
import seaborn as sns
import matplotlib.pyplot as plt

# Sentiment
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Sentiment Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Theme
plt.figure(figsize=(6,5))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens")
plt.title("Theme Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

