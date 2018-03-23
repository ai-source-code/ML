import matplotlib.pyplot as plt
def plotData(x, y):
    plt.scatter(x,y, marker='x', color='r', label='Training data')
    plt.xlabel("Population of city in 10,000")
    plt.ylabel("Profits in $10,000s")
    plt.xticks([5, 10, 15, 20, 25])
    plt.yticks([-5, 0, 5, 10, 15, 20, 25])
    plt.legend(loc='lower right')
    # plt.show(block = False)