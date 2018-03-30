import matplotlib.pyplot as plt
def plot_data(X, y):
    plt.scatter(X[:,0], X[:,1], X[:,2], y, linewidth=4)
    plt.title("Wine Quality")
    plt.show()