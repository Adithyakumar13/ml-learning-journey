import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize(x):
    meanx = x.sum() / m
    stdx = x.std()
    x = (x - meanx) / stdx
    return x, meanx, stdx

def gradient_decent(x, y, w, b, alpha = 0.001, loop = 100):

    cost_history = []
    prev_cost = float('inf')
    while(loop):
        y_cap = w * x + b
        error = y_cap - y

        temp_w = (1/m) * (error*x).sum()
        temp_b = (1/m) * error.sum()

        w = w - alpha * temp_w
        b = b - alpha * temp_b

        y_cap = w * x + b
        error = y_cap - y

        cost = (1 / (2*m)) * ((error**2).sum())
        cost_history.append(cost)
        
        if abs(prev_cost-cost) < (10**(-6)):
            print("min cost reached ")
            break

        prev_cost = cost

    return w, b, cost_history 

def denormalize(w, b, meanx, stdx):
    tempw = w
    w = w /stdx
    b = b - (tempw * meanx) / stdx
    return w,b

def display_model(w,b,df):
    #plotting points
    plt.scatter(df['SquareFeet'],df['Price'])
    plt.xlabel("Sq feet")
    plt.ylabel("Price")

    #plotting model
    points = np.linspace(df['SquareFeet'].min(), df['SquareFeet'].max(), 400)
    eq = w*points + b
    plt.plot(points,eq,color='red')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("./housing_price_dataset.csv")
    df = df[['Price','SquareFeet']]

    m = len(df) 
    w = 0.0
    b = 0.0

    x = df['SquareFeet'] 
    y = df['Price']

    x, meanx, stdx = normalize(x)

    w,b,cost_history = gradient_decent(x, y, w, b)
    print(f"The equation of line normalized {w}x + {b}")
    print("The cost history: ")
    print(f"{i}\t" for i in cost_history)
    
    w,b = denormalize(w, b, meanx, stdx)
    print(f"equateion of line denormalized: {w}x + {b}")

    display_model(w,b,df)