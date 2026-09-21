import numpy as np

RISK_FEATURES = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "Prefix_Suffix"]


def scores(X):
    """Fixed pre-registered risk score; URL_Length=0 never adds a point."""
    return (X[RISK_FEATURES].to_numpy() == -1).sum(axis=1) / len(RISK_FEATURES)
