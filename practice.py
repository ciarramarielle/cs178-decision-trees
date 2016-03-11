import numpy as np
np.random.seed(0)
import mltools as ml
import matplotlib.pyplot as plt
import mltools.dtree as dt
#reload(dt)

curve= np.genfromtxt("data/curve80.txt", delimiter=None)
X = curve[:,0]
X = X[:,np.newaxis]
Y = curve[:,1]

Xt, Xv, Yt, Yv = ml.splitData(X,Y,0.75)

lr = dt.treeRegress(Xt,Yt, maxDepth=20)


def plotMaxDepth_16():
    train_error = np.zeros(shape = (16,1))
    valid_error = np.zeros(shape = (16,1))
    depth = np.zeros(shape = (16,1))

    for i in range (0,16):
      lr = ml.dtree.treeRegress(Xt, Yt, maxDepth = i)
      train_error[i] = lr.mse(Xt, Yt)
      valid_error[i] = lr.mse(Xv, Yv)
      depth[i] = i


    plt.plot(depth, train_error, 'b', depth, valid_error, 'g')
    plt.legend (['train', 'validation'])

    plt.xlabel('depth')
    plt.ylabel('MSE')

    plt.show()

def plotMinParent_13():
    #minParent
    ertc = []
    ervc = []
    minparents = []

    for i in range(3,13):
        lr = ml.dtree.treeRegress(Xt, Yt, maxDepth = 20, minParent = 2**i)
        ertc.append( lr.mse(Xt, Yt))
        ervc.append( lr.mse(Xv, Yv))
        minparents.append(2**i)

    plt.plot(minparents, ertc , 'b', minparents, ervc , 'g')
    plt.legend(['train', 'validation'])

    plt.xlabel('minParent')
    plt.ylabel('MSE')

    plt.show()


if __name__ == "__main__":
    print("Exploring maxDepth")
    plotMaxDepth_16()
    #TODO: based on plot/errors, pick a max depth.

    print("Exploring minParent")
    plotMinParent_13()
    #TODO: based on plot/errors, pick a minParent

    #TODO: Explore minScore
    #TODO: Explore nFeatures