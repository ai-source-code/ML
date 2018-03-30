import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

def plot_multiple_features(X, y):
   sns.set(style="white", color_codes=True)
   features = pd.read_csv("winequality-red.csv")
#    print(features["quality"].value_counts())
#    features.plot(kind="scatter", x="density", y="quality")
   sns.pairplot(data=features, hue="quality", size=3)
#    sns.jointplot(x="density", y="quality", data=features, size=5)
   plt.show()
#    colors = ['b', 'c', 'y', 'm', 'r']
#    plt.scatter(X[:,0], X[:,1], X[:,2], y, linewidth=4)
#    plt.scatter(X[:,0], y, marker='x', color=colors[0])
#    plt.scatter(X[:,1], y, marker='o', color=colors[0])
#    plt.scatter(X[:,2], y, marker='o', color=colors[2])
#    plt.scatter(X[:,3], y, marker='o', color=colors[3])
#    plt.scatter(X[:,4], y, marker='o', color=colors[4])
#    plt.scatter(X[:,5], y, marker='o', color=colors[0])
#    plt.scatter(X[:,6], y, marker='o', color=colors[1])
#    plt.scatter(X[:,7], y, marker='o', color=colors[2])
#    plt.scatter(X[:,8], y, marker='o', color=colors[3])
#    plt.scatter(X[:,9], y, marker='x', color=colors[4])
#    colors=['b', 'c', 'y', 'm', 'r']

#    ax = plt.subplot(111, projection='3d')
#    plt.scatter(X[:,0], X[:,1], y, 'x', color=colors[0], label='Low Outlier')

#    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

#    plt.title("Wine Quality")
#    plt.show()