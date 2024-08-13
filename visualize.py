import matplotlib.pyplot as plt


def scatter(X, y, features):
    fig,ax = plt.subplots(1,5,figsize = (12,4),sharey = True)
    for i in range(len(ax)):
        ax[i].scatter(X[:,i],y,marker='.',color='g')
        ax[i].set_xlabel(features[i])
    ax[0].set_ylabel('Price (in 1000s)')
    plt.show()


def plot_cost_function(J_hist, iter):
    """
    Plot the cost function over the iterations.
    Args:
        J_hist: list of cost functions recorded during iterations
        iter: number of iterations
    """
    plt.figure(figsize=(8,6))
    plt.plot(range(0, min(iter, 1000)), J_hist, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.title('Cost Function vs. Iterations')
    plt.grid(True)
    plt.savefig('cost_function_gradient_descent.png')
    plt.show()