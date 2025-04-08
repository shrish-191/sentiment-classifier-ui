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
import gradio as gr
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Load model and tokenizer from Hugging Face
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

# Manually define the correct mapping
LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
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


