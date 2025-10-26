import pandas as pd

def preprocess_data(df):
    """
    Performs data cleaning, column dropping, and all encoding steps 
    (binary and one-hot encoding) as these transform individual columns, 
    adhering to the user's definition of preprocessing.
    """
    # 1. Removing irrelevant columns (df = df.drop(["LoanID"], axis = 1))

    if "LoanID" in df.columns:
        df = df.drop(["LoanID"], axis=1)

    df = df.dropna()
    
    # 2. Binary Encoding 
    with pd.option_context('future.no_silent_downcasting', True):
        df[["HasMortgage", "HasDependents", "HasCoSigner"]] = df[["HasMortgage", "HasDependents", "HasCoSigner"]].replace({"Yes": 1, "No": 0})
    
    # 3. One-Hot Encoding
    df = pd.get_dummies(df, columns = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"], dtype = int)
    
    return df