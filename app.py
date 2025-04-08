import gradio as gr
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


