import matplotlib.pyplot as plt

def Draw(feature1, feature2, poi, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "r", "k", "m", "g"]
    for f1, f2, poi_i in zip(feature1, feature2, poi):
        if poi_i :
            plt.scatter(f1, f2, color = colors[0], alpha=0.5)
        else:
            plt.scatter(f1, f2, color = colors[1], alpha=0.5)

    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


def Draw_outliers(feature1, feature2, y_predict, labels, f1_name="feature 1", f2_name="feature 2"):

    colors = ["b", "r", "k", "m", "g"]
    for f1, f2, outlier, label in zip(feature1, feature2, y_predict, labels):
        if outlier == 1:
            plt.scatter(f1, f2, color=colors[0])
        elif label == 0:
            plt.scatter(f1, f2, color=colors[1])
        else:
            plt.scatter(f1, f2, color=colors[2])
    plt.title('Red points are outliers')
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.show()