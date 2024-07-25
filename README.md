# Multiple Linear Regression from Scratch

This project demonstrates the implementation of multiple linear regression using only NumPy and Pandas, without relying on external machine learning libraries like scikit-learn. The project includes loading and preprocessing data, performing matrix operations to compute the regression coefficients, making predictions, and evaluating the model using the cost function.


## Formulas Used

### Multiple Linear Regression Model

The multiple linear regression model is represented as:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n $$

### Normal Equation

$$ y = \beta X $$

The regression coefficients are computed using the normal equation:

$$ \beta = (X^T X)^{-1} X^T y $$

### Cost Function

The cost function used to evaluate the model is the mean squared error:

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (X \beta - y)^2 $$

## Method Followed

1. **Loading and Preprocessing Data**:
   - Data is loaded from a CSV file.
   - Categorical data is converted to numerical values.
   - A column of ones is added to the feature matrix to account for the intercept term.

2. **Computing Regression Coefficients**:
   - The normal equation is used to compute the regression coefficients.

3. **Making Predictions**:
   - The model uses the computed coefficients to make predictions on the dataset.

4. **Evaluating the Model**:
   - The cost function is computed to evaluate the model's performance.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/linear_regression_project.git

