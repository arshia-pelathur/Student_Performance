
def gradient_descent(X, y, w_initial, b_initial, cost_function, gradients, lr, iter):
    """
    To perform Batch gradient descent to learn weights and parameter b. Updates them within in iter number of iterations and the set alpha.
    
    Args:
    X (array (m,n) ) : X training data
    y (array (m,) ) : target variable
    w_initial (n,) : initial weight matrix having a shape of (n,) where n is the number of features
    b_inital : initial constant value for the intercept
    cost_function : Cost function of the multiple linear regression equation for the weights,w and b 
    gradients : computes the gradients with respect to weights and parameter
    lr : alpha or learning rate
    iter : number of iterations
    
    Returns:
    w_final : final weights of all the features
    b_final : final value for constant parameter
    J_hist : all cost functions saved in this list
    """

    J_hist = []
    w_final = w_initial
    b_final = b_initial

    for i in range(iter):
        djdw, djdb = gradients(X, y, w_final, b_final)
        w_final = w_final - lr * djdw
        b_final = b_final - lr * djdb

        if i < 1000:
            J_hist.append(cost_function(X, y ,w_final, b_final))

        if i%100 == 0:
            print(f'\niteration {i} : Cost Function : {J_hist[-1]:0.3f} | ',
                  f'w : {w_final} | b : {b_final}')
    return w_final, b_final, J_hist