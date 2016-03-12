# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))

# USE IRIS DATA
iris= np.genfromtxt("data/iris.txt", delimiter=None)
X = iris[:,0:-1]
y = iris[:,-1]

print(np.shape(X))
print(np.shape(y))

# print("shape", np.shape(y))
#
# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
#
#


#REGRESSION TREE FOR MAX_DEPTH
def explore_max_depth():
    mse_list = []
    for i in range(1,16):
        lr = DecisionTreeRegressor(max_depth=i)
        lr.fit(X,y)
        # X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        y_1 = lr.predict(X)
        mse_list.append(mean_squared_error(y,y_1))
        # mse= mean

    l = [i for i in range(1,16)]
    plt.plot(l, mse_list, c="g", label="max_depth", linewidth=2)
    plt.show()
    pass


def explore_min_sample_split():
# #REGRESSION TREE FOR MIN_SAMPLE_SPLIT
    mse_list = []
    for i in range(3,13):
        lr = DecisionTreeRegressor(min_samples_split =2**i)
        lr.fit(X,y)
        # X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        y_1 = lr.predict(X)
        mse_list.append(mean_squared_error(y,y_1))
        # mse= mean

    l = [i for i in range(3,13)]
    plt.plot(l, mse_list, c="g", label="min_samples_split", linewidth=2)
    plt.show()
    pass


def explore_min_weight_fraction_leaf():
# #REGRESSION TREE FOR min_weight_fraction_leaf
    mse_list = []
    r = [0,0.1,0.2,0.3,0.4,0.5]
    for i in r:
        # i+=0.1
        # r.append(i)
        lr = DecisionTreeRegressor(min_weight_fraction_leaf =i)
        lr.fit(X,y)
        # X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        y_1 = lr.predict(X)
        mse_list.append(mean_squared_error(y,y_1))
        # mse= mean

    # l = [i for i in range(0,0.5)]
    plt.plot(r, mse_list, c="g", label="min_weight_fraction_leaf", linewidth=2)
    plt.show()
    pass

def presort_():
    mse_list = []
    lr_t=DecisionTreeRegressor(presort=True)
    lr_t.fit(X,y)
    y_t = lr_t.predict(X)
    mse_list.append(mean_squared_error(y,y_t))



    lr_f=DecisionTreeRegressor(presort=False)
    lr_f.fit(X,y)
    y_f = lr_f.predict(X)
    mse_list.append(mean_squared_error(y,y_f))
    plt.plot([0,1],mse_list, c="g", label="presort", linewidth=2)
    plt.show()


def explore_max_features():
    mse_list = []

    for i in range(1,5):
        lr = DecisionTreeRegressor(max_features =i)
        lr.fit(X,y)
        y_1= lr.predict(X)
        mse_list.append(mean_squared_error(y,y_1))

    plt.plot([1,2,3,4],mse_list, c="g", label="max_features", linewidth=2)
    plt.show()

    pass

if __name__=="__main__":
    print("USING SCIKIT.")
    explore_max_depth()
    explore_min_sample_split()
    explore_min_weight_fraction_leaf()

    #TODO: max_features is broken right now :'(
    # explore_max_features()