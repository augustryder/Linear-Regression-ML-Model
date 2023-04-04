import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")

# gets data for bmi and charges
X = data['bmi']
Y = data['charges']

# groups data by smokers
dfX_s = X.groupby(data["smoker"]).get_group('yes')
dfY_s = Y.groupby(data["smoker"]).get_group('yes')

# groups data for non-smokers
dfX_ns = X.groupby(data["smoker"]).get_group('no')
dfY_ns = Y.groupby(data["smoker"]).get_group('no')

# turns panda dataframes to lists
X = X.tolist()
Y = Y.tolist()

X_s = dfX_s.tolist()
Y_s = dfY_s.tolist()

X_ns = dfX_ns.tolist()
Y_ns = dfY_ns.tolist()

# cost function which returns a number that tells you how much error is in your line, lower number = more accurate
def cost_function (X, Y, w, b):
    sum_error = 0.0
    N = len(X)
    for i in range(N):
        sum_error += (w*X[i] + b - Y[i])**2
    return sum_error/(2*N)

# algorithm for getting new w and b values which result in a lower cost
# alpha = learning rate, higher learning rate will result in greater change in w and b but could
def gradient_descent(X, Y, w, b, alpha):
    dC_dw = 0.0
    dC_db = 0.0
    N = len(X)

    for i in range(N):
        dC_dw += (w*X[i] + b - Y[i])*X[i]
        dC_db += (w*X[i] + b - Y[i])

    dC_dw *= (1/N)
    dC_db *= (1/N)

    w = w - alpha*(dC_dw)
    b = b - alpha*(dC_db)

    return w, b

# implements gradient descent n number of times to get w and b values that result in the lowest cost
def train(X, Y, w, b, alpha, n):
    for i in range(n):
        w, b = gradient_descent(X, Y, w, b, alpha)

        # prints cost function every 1000 iterations of gd
        if i % 1000 == 0:
            print("iteration:", i, "cost: ", cost_function(X,Y,w,b))

    return w, b

# used to y value in dataset
def predict(x, w, b):
    return x*w + b

# displays scatterplot of data
sns.scatterplot(data = data, x = 'bmi', y = 'charges', hue='smoker', style='sex')
plt.title('Insurance cost vs. BMI of smokers and non-smokers')
plt.xlabel('BMI')
plt.ylabel('Charges (USD)')

# runs gradient descent 10000 times for smokers and non-smokers
w_s, b_s = train(X_s, Y_s, 0.0, 0.0, 0.0001, 10000)
w_ns, b_ns = train(X_ns, Y_ns, 0.0, 0.0, 0.0001, 10000)

# plots linear regression for smokers
y_s = w_s * dfX_s + b_s
sns.lineplot(x=X_s, y=y_s, color='blue')

# plots linear regression for non-smokers
y_ns = w_ns * dfX_ns + b_ns
sns.lineplot(x=X_ns, y=y_ns, color='red')

plt.show()