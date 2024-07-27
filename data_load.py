import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_preprocess(filepath):
    data = pd.read_csv(filepath)

    def label(val):
        if val == 'No': return 0
        else: return 1
    
    data['Extracurricular Activities'] = data['Extracurricular Activities'].apply(label)

    # Training Dataset
    X = np.array([
        data['Hours Studied'].values,
        data['Previous Scores'].values,
        data['Extracurricular Activities'].values,
        data['Sleep Hours'].values,
        data['Sample Question Papers Practiced'].values
    ]).T
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Target feature
    y = np.array(data['Performance Index'])

    # array of 1s for intercept
    X_ones = np.ones((X.shape[0],1))
    X = np.concatenate((X,X_ones),axis=1)       # training data along with intercept 
    return X, y