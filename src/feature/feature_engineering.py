import pandas as pd

def create_features(df):
    """
    Creates new, derivative features from existing columns:
    - LoanAmount/Income
    - LoanAmount/LoanTerm
    """
    if "LoanAmount" in df.columns and "Income" in df.columns:
        df["LoanAmount/Income"] = df["LoanAmount"] / (df["Income"] + 1e-6)

    if "LoanAmount" in df.columns and "LoanTerm" in df.columns:
        df["LoanAmount/LoanTerm"] = df["LoanAmount"] / (df["LoanTerm"] + 1e-6)
    
    return df