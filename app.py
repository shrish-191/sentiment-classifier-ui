'''import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Load model and tokenizer from your HF model repo
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    predictions = model(inputs).logits
    label = tf.argmax(predictions, axis=1).numpy()[0]
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[label]

demo = gr.Interface(fn=classify_sentiment,
                    inputs=gr.Textbox(placeholder="Enter a tweet..."),
                    outputs="text",
                    title="Tweet Sentiment Classifier",
                    description="Multilingual BERT-based Sentiment Analysis")

demo.launch()
'''
'''
import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Load model and tokenizer from Hugging Face
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

# Manually define the correct mapping
LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)
    pred_label = tf.argmax(probs, axis=1).numpy()[0]
    confidence = float(tf.reduce_max(probs).numpy())
    return f"Prediction: {LABELS[pred_label]} (Confidence: {confidence:.2f})"

demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(placeholder="Type your tweet here..."),
    outputs="text",
    title="Sentiment Analysis on Tweets",
    description="Multilingual BERT model fine-tuned for sentiment classification. Labels: Positive, Neutral, Negative."
)

demo.launch()
'''
'''
import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import snscrape.modules.twitter as sntwitter
import praw
import os

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

# Label Mapping
LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

# Reddit API setup with environment variables
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-script")
)

# Tweet text extractor
def fetch_tweet_text(tweet_url):
    try:
        tweet_id = tweet_url.split("/")[-1]
        for tweet in sntwitter.TwitterTweetScraper(tweet_id).get_items():
            return tweet.content
        return "Unable to extract tweet content."
    except Exception as e:
        return f"Error fetching tweet: {str(e)}"

# Reddit post extractor
def fetch_reddit_text(reddit_url):
    try:
        submission = reddit.submission(url=reddit_url)
        return f"{submission.title}\n\n{submission.selftext}"
    except Exception as e:
        return f"Error fetching Reddit post: {str(e)}"

# Sentiment classification logic
def classify_sentiment(text_input, tweet_url, reddit_url):
    if reddit_url.strip():
        text = fetch_reddit_text(reddit_url)
    elif tweet_url.strip():
        text = fetch_tweet_text(tweet_url)
    elif text_input.strip():
        text = text_input
    else:
        return "[!] Please enter text or a post URL."

    if text.lower().startswith("error") or "Unable to extract" in text:
        return f"[!] Error: {text}"

    try:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1)
        pred_label = tf.argmax(probs, axis=1).numpy()[0]
        confidence = float(tf.reduce_max(probs).numpy())
        return f"Prediction: {LABELS[pred_label]} (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"[!] Prediction error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=[
        gr.Textbox(label="Custom Text Input", placeholder="Type your tweet or message here..."),
        gr.Textbox(label="Tweet URL", placeholder="Paste a tweet URL here (optional)"),
        gr.Textbox(label="Reddit Post URL", placeholder="Paste a Reddit post URL here (optional)")
    ],
    outputs="text",
    title="Multilingual Sentiment Analysis",
    description="Analyze sentiment of text, tweets, or Reddit posts. Supports multiple languages using BERT!"
)

demo.launch()
'''

 
'''
import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import praw
import os


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax


model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}


fallback_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
fallback_model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)

# Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-ui")
)

def fetch_reddit_text(reddit_url):
    try:
        submission = reddit.submission(url=reddit_url)
        return f"{submission.title}\n\n{submission.selftext}"
    except Exception as e:
        return f"Error fetching Reddit post: {str(e)}"


def fallback_classifier(text):
    encoded_input = fallback_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = fallback_model(**encoded_input)
    scores = softmax(output.logits.numpy()[0])
    labels = ['Negative', 'Neutral', 'Positive']
    return f"Prediction: {labels[scores.argmax()]}"

def classify_sentiment(text_input, reddit_url):
    if reddit_url.strip():
        text = fetch_reddit_text(reddit_url)
    elif text_input.strip():
        text = text_input
    else:
        return "[!] Please enter some text or a Reddit post URL."

    if text.lower().startswith("error") or "Unable to extract" in text:
        return f"[!] {text}"

    try:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1)
        confidence = float(tf.reduce_max(probs).numpy())
        pred_label = tf.argmax(probs, axis=1).numpy()[0]

        if confidence < 0.5:
            return fallback_classifier(text)

        return f"Prediction: {LABELS[pred_label]}"
    except Exception as e:
        return f"[!] Prediction error: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=[
        gr.Textbox(
            label="Text Input (can be tweet or any content)",
            placeholder="Paste tweet or type any content here...",
            lines=4
        ),
        gr.Textbox(
            label="Reddit Post URL",
            placeholder="Paste a Reddit post URL (optional)",
            lines=1
        ),
    ],
    outputs="text",
    title="Sentiment Analyzer",
    description="🔍 Paste any text (including tweet content) OR a Reddit post URL to analyze sentiment.\n\n💡 Tweet URLs are not supported directly due to platform restrictions. Please paste tweet content manually."
)

demo.launch()
'''
'''
import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import praw
import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

# Install tesseract OCR (only runs once in Hugging Face Spaces)
os.system("apt-get update && apt-get install -y tesseract-ocr")

# Load main model
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

# Load fallback model
fallback_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
fallback_model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-ui")
)

def fetch_reddit_text(reddit_url):
    try:
        submission = reddit.submission(url=reddit_url)
        return f"{submission.title}\n\n{submission.selftext}"
    except Exception as e:
        return f"Error fetching Reddit post: {str(e)}"

def fallback_classifier(text):
    encoded_input = fallback_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = fallback_model(**encoded_input)
    scores = softmax(output.logits.numpy()[0])
    labels = ['Negative', 'Neutral', 'Positive']
    return f"Prediction: {labels[scores.argmax()]}"

def clean_ocr_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces and newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text

def classify_sentiment(text_input, reddit_url, image):
    if reddit_url.strip():
        text = fetch_reddit_text(reddit_url)
    elif image is not None:
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh)
            text = clean_ocr_text(text)
        except Exception as e:
            return f"[!] OCR failed: {str(e)}"
    elif text_input.strip():
        text = text_input
    else:
        return "[!] Please enter some text, upload an image, or provide a Reddit URL."

    if text.lower().startswith("error") or "Unable to extract" in text:
        return f"[!] {text}"

    try:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=1)
        confidence = float(tf.reduce_max(probs).numpy())
        pred_label = tf.argmax(probs, axis=1).numpy()[0]

        if confidence < 0.5:
            return fallback_classifier(text)

        return f"Prediction: {LABELS[pred_label]}"
    except Exception as e:
        return f"[!] Prediction error: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=[
        gr.Textbox(
            label="Text Input (can be tweet or any content)",
            placeholder="Paste tweet or type any content here...",
            lines=4
        ),
        gr.Textbox(
            label="Reddit Post URL",
            placeholder="Paste a Reddit post URL (optional)",
            lines=1
        ),
        gr.Image(
            label="Upload Image (optional)",
            type="pil"
        )
    ],
    outputs="text",
    title="Sentiment Analyzer",
    description="🔍 Paste any text, Reddit post URL, or upload an image containing text to analyze sentiment.\n\n💡 Tweet URLs are not supported. Please paste tweet content or screenshot instead."
)

demo.launch()
'''

import gradio as gr
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import praw
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
import os

# Load pre-trained model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = AutoTokenizer.from_pretrained("shrish191/sentiment-bert")

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Sentiment Prediction Function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)[0]
    probs = tf.nn.softmax(outputs, axis=1).numpy()
    pred_label = np.argmax(probs, axis=1)[0]
    return label_map[pred_label]

# Reddit URL Handling
def analyze_reddit_url(url):
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="YOUR_USER_AGENT"
    )
    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list() if len(comment.body) > 10][:100]
        sentiments = [predict_sentiment(comment) for comment in comments]
        sentiment_counts = Counter(sentiments)
        result_text = "\n".join([f"{s}: {c}" for s, c in sentiment_counts.items()])

        # Pie chart
        fig = go.Figure(data=[go.Pie(labels=list(sentiment_counts.keys()),
                                     values=list(sentiment_counts.values()),
                                     hole=0.3)])
        fig.update_layout(title="Sentiment Distribution of Reddit Comments")
        return result_text, fig
    except Exception as e:
        return str(e), None

# Subreddit Analysis Function
def analyze_subreddit(subreddit_name):
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="YOUR_USER_AGENT"
    )
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = list(subreddit.hot(limit=100))
        texts = [post.title + " " + post.selftext for post in posts if post.selftext or post.title]

        if not texts:
            return "No valid text data found in subreddit.", None

        sentiments = [predict_sentiment(text) for text in texts]
        sentiment_counts = Counter(sentiments)
        result_text = "\n".join([f"{s}: {c}" for s, c in sentiment_counts.items()])

        # Pie chart
        fig = go.Figure(data=[go.Pie(labels=list(sentiment_counts.keys()),
                                     values=list(sentiment_counts.values()),
                                     hole=0.3)])
        fig.update_layout(title=f"Sentiment Distribution in r/{subreddit_name}")

        return result_text, fig
    except Exception as e:
        return str(e), None

# Image Upload Functionality
from PIL import Image
import pytesseract

def extract_text_from_image(image):
    try:
        img = Image.open(image)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def analyze_image_sentiment(image):
    extracted_text = extract_text_from_image(image)
    if extracted_text:
        sentiment = predict_sentiment(extracted_text)
        return f"Extracted Text: {extracted_text}\n\nPredicted Sentiment: {sentiment}"
    return "No text extracted."

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Sentiment Analysis App")
    with gr.Tab("Analyze Text"):
        input_text = gr.Textbox(label="Enter text")
        output_text = gr.Textbox(label="Predicted Sentiment")
        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(fn=predict_sentiment, inputs=input_text, outputs=output_text)

    with gr.Tab("Analyze Reddit URL"):
        reddit_url = gr.Textbox(label="Enter Reddit post URL")
        url_result = gr.Textbox(label="Sentiment Counts")
        url_plot = gr.Plot(label="Pie Chart")
        analyze_url_btn = gr.Button("Analyze Reddit Comments")
        analyze_url_btn.click(fn=analyze_reddit_url, inputs=reddit_url, outputs=[url_result, url_plot])

    with gr.Tab("Analyze Image"):
        image_input = gr.Image(label="Upload an image")
        image_result = gr.Textbox(label="Sentiment from Image Text")
        analyze_img_btn = gr.Button("Analyze Image")
        analyze_img_btn.click(fn=analyze_image_sentiment, inputs=image_input, outputs=image_result)

    with gr.Tab("Analyze Subreddit"):
        subreddit_input = gr.Textbox(label="Enter subreddit name (without r/)")
        subreddit_result = gr.Textbox(label="Sentiment Counts")
        subreddit_plot = gr.Plot(label="Pie Chart")
        analyze_subreddit_btn = gr.Button("Analyze Subreddit")
        analyze_subreddit_btn.click(fn=analyze_subreddit, inputs=subreddit_input, outputs=[subreddit_result, subreddit_plot])

demo.launch()






