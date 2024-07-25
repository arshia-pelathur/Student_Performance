import numpy as np
import pandas as pd
from data_load import load_preprocess
from model import MultipleLinearRegression
from cost_functions import compute_cost

def main():
    X, y = load_preprocess('Student_Performance.csv')

    model = MultipleLinearRegression()
    model.fit(X,y)

    weights = model.get_weights()

    predictions = model.predict(X)

    cost = compute_cost(predictions,y)
    print(f'Multiple Linear Regression Equation:\n\t Y  = {weights[0]:.2f} X1  +  {weights[1]:.2f} X2  +  {weights[2]:.2f} X3  +  {weights[3]:.2f} X4  +  {weights[4]:.2f} X5  +  {weights[5]:.2f}')
    print('\nCost Function of above Equation: ',f'{cost:.4f}')

    comparison = pd.DataFrame({'Actual': y, 'Predicted': predictions})
    print(comparison.head())
    
if __name__ == '__main__':
    print('\n___________________________MULTIPLE LINEAR REGRESSION__________________________\n')
    main()