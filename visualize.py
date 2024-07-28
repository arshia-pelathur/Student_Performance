import matplotlib.pyplot as plt


def scatter(X, y, features):
    fig,ax = plt.subplots(1,5,figsize = (15,4),sharey = True)
    for i in range(len(ax)):
        ax[i].scatter(X[:,i],y,marker='.',color='r')
        ax[i].set_xlabel(features[i])
    ax[0].set_ylabel('Price (in 1000s)')
    plt.show()