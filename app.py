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
import matplotlib.pyplot as plt
import pandas as pd

# Install tesseract OCR (only runs once in Hugging Face Spaces)
os.system("apt-get update && apt-get install -y tesseract-ocr")

# Load main model
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}

# Load fallback model
fallback_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
fallback_model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-ui-finalyear-shrish191")
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
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
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

# Subreddit sentiment analysis function
def analyze_subreddit(subreddit_name):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = list(subreddit.hot(limit=20))

        sentiments = []
        titles = []

        for post in posts:
            text = f"{post.title}\n{post.selftext}"
            try:
                inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
                outputs = model(inputs)
                probs = tf.nn.softmax(outputs.logits, axis=1)
                confidence = float(tf.reduce_max(probs).numpy())
                pred_label = tf.argmax(probs, axis=1).numpy()[0]

                sentiment = LABELS[pred_label] if confidence >= 0.5 else fallback_classifier(text).split(": ")[-1]
            except:
                sentiment = "Error"
            sentiments.append(sentiment)
            titles.append(post.title)

        df = pd.DataFrame({"Title": titles, "Sentiment": sentiments})
        sentiment_counts = df["Sentiment"].value_counts()

        # Plot bar chart
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", color=["red", "green", "gray"], ax=ax)
        ax.set_title(f"Sentiment Distribution in r/{subreddit_name}")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Posts")

        return fig, df
    except Exception as e:
        return f"[!] Error: {str(e)}", pd.DataFrame()

# Gradio tab 1: Text/Image/Reddit Post Analysis
main_interface = gr.Interface(
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

# Gradio tab 2: Subreddit Analysis
subreddit_interface = gr.Interface(
    fn=analyze_subreddit,
    inputs=gr.Textbox(label="Subreddit Name", placeholder="e.g., AskReddit"),
    outputs=[
        gr.Plot(label="Sentiment Distribution"),
        gr.Dataframe(label="Post Titles and Sentiments", wrap=True)
    ],
    title="Subreddit Sentiment Analysis",
    description="📊 Enter a subreddit to analyze sentiment of its top 20 hot posts."
)

# Tabs
demo = gr.TabbedInterface(
    interface_list=[main_interface, subreddit_interface],
    tab_names=["General Sentiment Analysis", "Subreddit Analysis"]
)

demo.launch()






















 

 

  

  










