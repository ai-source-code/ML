import matplotlib.pyplot as plt
def plotData(x, y):
    plt.scatter(x,y, marker='x', color='r')
    plt.xlabel("Population of city in 10,000")
    plt.ylabel("Profits in $10,000s")
    # plt.show(block = False)