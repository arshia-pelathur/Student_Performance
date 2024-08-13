import numpy as np
import pandas as pd
from data_load import load_preprocess
from model import MultipleLinearRegression
from cost_functions import compute_cost
from computing import gradient, cost_function
from algorithm import gradient_descent
from visualize import scatter, plot_cost_function

def main():
    X, y = load_preprocess('Student_Performance.csv')

    model = MultipleLinearRegression()
    model.fit(X,y)

    weights = model.get_weights()

    predictions = model.predict(X)

    cost = compute_cost(predictions,y)
    print(f'Multiple Linear Regression Equation:\n\t Y  = {weights[0]:.2f} X1  +  {weights[1]:.2f} X2  +  {weights[2]:.2f} X3  +  {weights[3]:.2f} X4  +  {weights[4]:.2f} X5  +  {weights[5]:.2f}')
    print('\nCost of above Equation: ',f'{cost:.4f}')

    print('\nUsing Gradient Descent:')

    w_initial = np.zeros(X.shape[1]-1)        # subtracting 1 as weight matrix doesnt have any weights for the value of b which is all ones in the last columns.
    b_initial = 0
    iterations = 1000
    alpha = 0.01
    w_final, b_final, J_Hist = gradient_descent(X, y, w_initial, b_initial, cost_function, gradient, alpha, iterations)
    print(f"\n(Final w, Final b) found by gradient descent: ({w_final},{b_final:.4f})")
    print(f'\nMultiple Linear Regression using Gradient Descent Equation:\n\t Y  = {w_final[0]:.2f} X1  +  {w_final[1]:.2f} X2  +  {w_final[2]:.2f} X3  +  {w_final[3]:.2f} X4  +  {w_final[4]:.2f} X5  +  {b_final:.2f}')

    plot_cost_function(J_Hist,iterations)
    features = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']
    scatter(X,y,features)


if __name__ == '__main__':
    print('\n___________________________MULTIPLE LINEAR REGRESSION__________________________\n')
    main()