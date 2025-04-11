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
import gradio as gr
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
'''demo = gr.Interface(
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
)'''
demo = gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}")

with demo:
    gr.Markdown("""
    # 🌟 Sentiment Analysis Tool
    *Uncover the emotional tone behind text content and Reddit posts*
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📥 Input Options")
            with gr.Tabs():
                with gr.TabItem("📝 Text Content"):
                    text_input = gr.Textbox(
                        label="Paste Your Text Content",
                        placeholder="Enter tweet, comment, or any text here...",
                        lines=5,
                        elem_id="text-input"
                    )
                with gr.TabItem("🔗 Reddit Post"):
                    url_input = gr.Textbox(
                        label="Reddit Post URL",
                        placeholder="Paste Reddit post URL here (e.g., https://www.reddit.com/r/...)",
                        lines=1,
                        elem_id="url-input"
                    )
            gr.Markdown("""
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 10px;">
                ⚠️ Note: For Twitter analysis, please paste text content directly due to platform restrictions
            </div>
            """, elem_id="warning-box")
            
        with gr.Column():
            gr.Markdown("## 📊 Analysis Results")
            output_text = gr.Textbox(
                label="Sentiment Assessment",
                placeholder="Your analysis will appear here...",
                interactive=False,
                lines=5,
                elem_id="result-box"
            )
            examples = gr.Examples(
                examples=[
                    ["Just had the most amazing dinner! The service was incredible!"],
                    ["https://www.reddit.com/r/technology/comments/xyz123/new_ai_breakthrough"],
                    ["Really disappointed with the latest update. Features are missing and it's so slow."]
                ],
                inputs=[text_input],
                label="💡 Try These Examples"
            )

    gr.Markdown("""
    <div style="text-align: center; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
        🚀 Powered by Gradio & Hugging Face | 
        [Privacy Policy](#) | 
        [Terms of Service](#) | 
        [GitHub Repo](#)
    </div>
    """, elem_id="footer")


demo.launch()


