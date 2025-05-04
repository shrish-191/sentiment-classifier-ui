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
    fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
    fallback_model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)

    # Tokenize and predict
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    outputs = model(inputs)
    predictions = tf.math.argmax(outputs.logits, axis=1).numpy()

    # Generate report
    report = classification_report(true_labels, predictions, target_names=["negative", "neutral", "positive"])
    return report
