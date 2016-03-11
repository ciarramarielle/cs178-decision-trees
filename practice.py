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

Xt, Xe, Yt, Ye = ml.splitData(X,Y,0.75)

lr = dt.treeRegress(Xt,Yt, maxDepth=20)