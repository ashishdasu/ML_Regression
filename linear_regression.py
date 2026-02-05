'''
Ashish Dasu
CS6140

Linear Regression using closed form solution and gradient descent
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    '''
    Load training and testing data from CSV files.
    '''
    train_data = pd.read_csv('data/q7-train.csv')
    test_data = pd.read_csv('data/q7-test.csv')
    
    # Extract features and targets
    X_train = train_data['x'].values
    y_train = train_data['y'].values
    X_test = test_data['x'].values
    y_test = test_data['y'].values
    
    return X_train, y_train, X_test, y_test


def plot_data(X_train, y_train, X_test, y_test):
    '''
    Plot training and testing data points.
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c='blue', alpha=0.6, label='Training Data')
    plt.scatter(X_test, y_test, c='red', alpha=0.6, label='Testing Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training and Testing Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def construct_design_matrix(X, degree):
    '''
    Constructing design matrix for polynomial regression.
    
    Args:
        X: input features (1D array)
        degree: polynomial degree
    
    Returns:
        Design matrix with shape (N, degree+1)
        First column is all 1s (for intercept), then x, x^2, x^3, ...
    '''
    
    N = len(X)
    design_matrix = np.zeros((N, degree + 1))
    
    for i in range(degree + 1):
        design_matrix[:, i] = X ** i
    
    return design_matrix


def compute_mse(y_true, y_pred):
    '''
        Compute Mean Squared Error.
        
        MSE = (1/N) * sum((y_true - y_pred)^2)
    '''
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Plot data
    plot_data(X_train, y_train, X_test, y_test)