def predict(model, x, threshold):
    """Return phishing probability and alert decision for a feature row/batch."""
    probability = model.predict_proba(x)[:, 1]
    return probability, probability >= threshold
