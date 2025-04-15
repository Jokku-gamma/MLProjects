import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(0)
x=np.linspace(0,5,10).reshape(-1,1)
y=np.sin(x).ravel()
noise=np.array([0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.1, 0.1, -0.05])
y=y+noise

degrees=[1,3,6,10]
for d in degrees:
    model=make_pipeline(PolynomialFeatures(d),LinearRegression())
    model.fit(x,y)
    x_test=np.linspace(0,5,100).reshape(-1,1)
    y_pred=model.predict(x_test)

    plt.plot(x_test,y_pred,label=f"Degree {d}")
    plt.scatter(x,y,facecolor='red',label='Training Data')

plt.legend()
plt.title("Bias-Variance Tradeoff")
plt.xlabel("x")
plt.ylabel("y")
plt.show()