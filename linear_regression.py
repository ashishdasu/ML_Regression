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

def closed_form_solution(X, y):
    '''
    Compute closed-form solution for linear regression.
    
    Formula: theta = (X^T X)^(-1) X^T y
    
    Args:
        X: design matrix (N x (degree+1))
        y: target values (N,)
    
    Returns:
        theta: coefficients (degree+1,)
    '''
    # theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return theta
    
def gradient_descent(X, y, learning_rate=0.01, max_iterations=10000, tolerance=1e-6):
    '''
    Compute solution using gradient descent.
    
    Update rule: theta = theta - learning_rate * gradient
    Gradient: gradient = 2 * X^T * (X*theta - y)
    
    Args:
        X: design matrix (N x (degree+1))
        y: target values (N,)
        learning_rate: step size (eta)
        max_iterations: maximum number of iterations
        tolerance: convergence threshold
    
    Returns:
        theta: coefficients (degree+1,)
    '''
    # Initialize theta to zeros
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    
    for i in range(max_iterations):
        predictions = X @ theta
        
        # Compute gradient: 2 * X^T * (X*theta - y)
        gradient = 2 * X.T @ (predictions - y)

        theta_new = theta - learning_rate * gradient
        
        # Checking for convergence--if change in theta is small
        if np.linalg.norm(theta_new - theta) < tolerance:
            break
        
        theta = theta_new
    
    return theta


def train_and_evaluate(X_train, y_train, X_test, y_test, theta, method_name):
    '''
    Evaluate a trained model on train and test sets.
    
    Args:
        X_train, y_train: training data (design matrix form)
        X_test, y_test: testing data (design matrix form)
        theta: trained coefficients
        method_name: string name for printing
    
    Returns:
        Dictionary with predictions and metrics
    '''
    # Make predictions
    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta
    
    # Calculate MSE
    train_mse = compute_mse(y_train, y_train_pred)
    test_mse = compute_mse(y_test, y_test_pred)
    
    # Display results
    print(f'\n{method_name}')
    print('~' * 40)
    print(f'Theta: {theta}')
    print(f'Training MSE: {train_mse:.6f}')
    print(f'Testing MSE:  {test_mse:.6f}')
    
    return {
        'theta': theta,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }


def compare_methods(results_closed, results_gd):
    '''Compare results from closed-form and gradient descent.'''
    print('\nComparison')
    print('~' * 40)
    theta_diff = np.abs(results_closed['theta'] - results_gd['theta'])
    print(f'Theta difference: {theta_diff}')
    print(f'Max difference: {np.max(theta_diff):.8f}')


def run_linear_model(X_train, y_train, X_test, y_test, degree=1):
    '''
    Run both closed-form and gradient descent for a given polynomial degree.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: testing data
        degree: polynomial degree
    
    Returns:
        Dictionary with results from both methods
    '''
    print(f'\n{"~"*60}')
    print(f'POLYNOMIAL DEGREE {degree}')
    print(f'{"~"*60}')
    
    # Build design matrices
    X_train_design = construct_design_matrix(X_train, degree)
    X_test_design = construct_design_matrix(X_test, degree)
    
    # Closed-form solution
    theta_closed = closed_form_solution(X_train_design, y_train)
    results_closed = train_and_evaluate(X_train_design, y_train, X_test_design, y_test, 
                                        theta_closed, 'Closed-Form Solution')
    
    # Gradient descent solution
    lr = 0.0001 / (degree ** 3)
    max_iter = 50000 * degree
    theta_gd = gradient_descent(X_train_design, y_train, learning_rate=lr, max_iterations=max_iter)
    results_gd = train_and_evaluate(X_train_design, y_train, X_test_design, y_test,
                                    theta_gd, 'Gradient Descent')
    
    # Compare the two methods
    compare_methods(results_closed, results_gd)
    
    return {
        'degree': degree,
        'closed': results_closed,
        'gd': results_gd
    }

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Part 2: Linear model (degree 1)
    results_linear = run_linear_model(X_train, y_train, X_test, y_test, degree=1)
    
    # Part 3: Quadratic model (degree 2)
    results_quadratic = run_linear_model(X_train, y_train, X_test, y_test, degree=2)
    
    # Part 4: Cubic model (degree 3)
    results_cubic = run_linear_model(X_train, y_train, X_test, y_test, degree=3)