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

# Load model and tokenizer from your HF model repo
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

def classify_sentiment(text):
    text = text.lower().strip()  # Normalize input
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    predictions = model(inputs).logits
    label = tf.argmax(predictions, axis=1).numpy()[0]
    labels = model.config.id2label  # Use mapping from config.json
    print(f"Text: {text} | Prediction: {label} | Logits: {predictions.numpy()}")  # Debug
    return labels[str(label)]  # Convert to string key

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

# Load model and tokenizer from Hugging Face Hub
model = TFBertForSequenceClassification.from_pretrained("shrish191/sentiment-bert")
tokenizer = BertTokenizer.from_pretrained("shrish191/sentiment-bert")

def classify_sentiment(text):
    text = text.lower().strip()
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(inputs, training=False)
    logits = outputs.logits
    label_id = int(tf.argmax(logits, axis=1).numpy()[0])
    
    # Handle label mapping correctly
    raw_labels = model.config.id2label
    if isinstance(list(raw_labels.keys())[0], str):
        label = raw_labels.get(str(label_id), "Unknown")
    else:
        label = raw_labels.get(label_id, "Unknown")
    
    print(f"Text: {text} | Label ID: {label_id} | Label: {label} | Logits: {logits.numpy()}")
    return label

# Define the Gradio interface
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(placeholder="Enter a tweet..."),
    outputs="text",
    title="Tweet Sentiment Classifier",
    description="Multilingual BERT-based Sentiment Analysis"
)

# Launch the app
demo.launch()

