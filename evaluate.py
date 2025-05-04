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

from evaluate import get_classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd
def get_classification_report():
    from sklearn.metrics import classification_report
    import pandas as pd

    # Load your test data
    df = pd.read_csv("test.csv")
    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    # Load tokenizer and model
    #tokenizer = AutoTokenizer.from_pretrained("Shrish/mbert-sentiment")
    #model = TFAutoModelForSequenceClassification.from_pretrained("Shrish/mbert-sentiment")
    fallback_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)

    # Tokenize and predict
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    outputs = model(inputs)
    predictions = tf.math.argmax(outputs.logits, axis=1).numpy()

    # Generate report
    report = classification_report(true_labels, predictions, target_names=["negative", "neutral", "positive"])
    return report
