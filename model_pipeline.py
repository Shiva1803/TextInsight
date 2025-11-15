
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_models():
    device = 0 if torch.cuda.is_available() else -1
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    theme_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=device
    )
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return sentiment_pipe, theme_pipe, summarizer, tokenizer, model


def analyze_single_text(text, sentiment_pipe, theme_pipe, candidate_labels):
    sent = sentiment_pipe(text)[0]
    theme = theme_pipe(text, candidate_labels)

    sentiment_label = sent["label"]
    sentiment_score = sent["score"]

    top_theme = theme["labels"][0]
    top_theme_score = theme["scores"][0]
    
    all_theme_scores = {label: score for label, score in zip(theme["labels"], theme["scores"])}

    return sentiment_label, sentiment_score, top_theme, top_theme_score, all_theme_scores


def analyze_batch(df, sentiment_pipe, theme_pipe, candidate_labels):
    df = df.copy()

    df["sentiment"] = df["text"].apply(lambda x: sentiment_pipe(x)[0]["label"])
    df["sentiment_score"] = df["text"].apply(lambda x: sentiment_pipe(x)[0]["score"])

    df["theme"] = df["text"].apply(lambda x: theme_pipe(x, candidate_labels)["labels"][0])
    df["theme_score"] = df["text"].apply(lambda x: theme_pipe(x, candidate_labels)["scores"][0])

    return df


def get_token_sentiment_scores(text, tokenizer, model):
    """
    Extract token-level sentiment contribution scores.
    Returns list of (token, score) tuples where:
    - Positive score = contributes to positive sentiment
    - Negative score = contributes to negative sentiment
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
        attentions = torch.stack(outputs.attentions).mean(dim=0)
        attentions = attentions.mean(dim=1)
        attentions = attentions[0].mean(dim=0)
        
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)
        
        pred_class = torch.argmax(logits).item()
        
        sentiment_direction = probs[1] - probs[0]
    

    token_scores = []
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        score = attentions[i].item() * sentiment_direction.item()
        
        clean_token = token.replace("##", "").replace("▁", "")
        token_scores.append((clean_token, score))
    
    return token_scores, pred_class


# Generate summary from multiple texts
def generate_summary(texts, summarizer, max_length=150, min_length=50):
    """
    Generate a summary from a list of texts.
    Combines texts and generates an overall summary.
    Safe for large CSVs - truncates to prevent BART from exploding.
    """

    joined = " ".join(texts)
    
    if len(joined) > 3000:
        joined = joined[:3000]
    
    try:
        result = summarizer(
            joined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]["summary_text"]
    except Exception as e:
        return f"Could not generate summary: {str(e)}"



from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_sentiment(df, sentiment_pipe):
    """
    Evaluate sentiment model using labeled dataset.
    CSV must contain columns: text, true_sentiment
    """
    df = df.copy()
    
    df["predicted"] = df["text"].apply(lambda x: sentiment_pipe(x)[0]["label"])

    accuracy = accuracy_score(df["true_sentiment"], df["predicted"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["true_sentiment"], df["predicted"], average="weighted"
    )
    cm = confusion_matrix(df["true_sentiment"], df["predicted"])
    report = classification_report(df["true_sentiment"], df["predicted"])

    return accuracy, precision, recall, f1, cm, report


def evaluate_theme(df, theme_pipe, candidate_labels):
    """
    Evaluate zero-shot theme classifier.
    CSV must contain: text, true_theme
    """
    df = df.copy()

    df["predicted"] = df["text"].apply(
        lambda x: theme_pipe(x, candidate_labels)["labels"][0]
    )

    # metrics
    accuracy = accuracy_score(df["true_theme"], df["predicted"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["true_theme"], df["predicted"], average="weighted"
    )
    cm = confusion_matrix(df["true_theme"], df["predicted"])
    report = classification_report(df["true_theme"], df["predicted"])

    return accuracy, precision, recall, f1, cm, report
