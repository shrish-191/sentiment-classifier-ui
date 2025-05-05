from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import classification_report
import tensorflow as tf
import pandas as pd

def get_classification_report():
    try:
        # Load test data
        df = pd.read_csv("test.csv")
        texts = df["text"].tolist()
        true_labels = df["label"].tolist()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("Shrish/mbert-sentiment")
        model = TFAutoModelForSequenceClassification.from_pretrained("Shrish/mbert-sentiment")

        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
        outputs = model(inputs)
        preds = tf.math.argmax(outputs.logits, axis=1).numpy()

        # Generate report
        report = classification_report(true_labels, preds, target_names=["negative", "neutral", "positive"])
        return report
    except Exception as e:
        return f"⚠️ Error occurred: {str(e)}"
