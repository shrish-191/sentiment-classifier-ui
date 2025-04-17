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
'''import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import praw
import os

# Load model and tokenizer from Hugging Face
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

# Label mapping
LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

# Reddit API setup (credentials loaded securely from secrets)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-script")
)

# Reddit post fetcher
def fetch_reddit_text(reddit_url):
    try:
        submission = reddit.submission(url=reddit_url)
        return f"{submission.title}\n\n{submission.selftext}"
    except Exception as e:
        return f"Error fetching Reddit post: {str(e)}"

# Main sentiment function
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
        pred_label = tf.argmax(probs, axis=1).numpy()[0]
        confidence = float(tf.reduce_max(probs).numpy())
        return f"Prediction: {LABELS[pred_label]} (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"[!] Prediction error: {str(e)}"

# Gradio UI
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
    user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-classifier-script")
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





