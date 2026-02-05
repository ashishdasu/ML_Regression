'''
Ashish Dasu
CS6140 - Machine Learning

Polynomial Regression with Regularization - Airfoil Noise Dataset
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

def load_airfoil_data():
    '''
    Load training, validation, and test data for airfoil noise prediction.
    Returns separate feature matrices and target vectors for each split.
    '''
    train = pd.read_csv('data/q8-train.csv')
    val = pd.read_csv('data/q8-val.csv')
    test = pd.read_csv('data/q8-test.csv')
    
    target_column = 'Scaled sound pressure level'
    
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    
    X_val = val.drop(columns=[target_column])
    y_val = val[target_column]
    
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_mse(y_true, y_pred):
    '''Calculate mean squared error between predictions and true values.'''
    return np.mean((y_true - y_pred) ** 2)

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_airfoil_data()
    
    print(f'Training samples: {len(X_train)}')
    print(f'Validation samples: {len(X_val)}')
    print(f'Testing samples: {len(X_test)}')
    print(f'\nFeatures: {list(X_train.columns)}')