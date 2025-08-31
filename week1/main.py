import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load data set to a data frame
df = pd.read_csv("./housing_price_dataset.csv")
df = df[['Price','SquareFeet']]

m = len(df) 
w = 0.0
b = 0.0
alpha = 0.001
loop = 100
x = df['SquareFeet'] 
y = df['Price']
prev = float('inf')

#normalizing data
meanx = x.sum() / m
meany = y.sum() / m

stdx = x.std()
stdy = y.std()

x = (x - meanx) / stdx
#y = (y - meany) / stdy


#gradient decent
while(True):
    y_cap = w * x + b
    error = y_cap - y

    temp_w = (1/m) * (error*x).sum()
    temp_b = (1/m) * error.sum()

    w = w - alpha * temp_w
    b = b - alpha * temp_b

    y_cap = w * x + b
    error = y_cap - y

    cost = (1 / (2*m)) * ((error**2).sum())
    
    if abs(prev-cost) < (10**(-6)):
        print("min cost reached ")
        break
    prev = cost
print(f"The equation of line normalized {w}x + {b}")

#denormalizing equation
tempw = w
w = w /stdx

b = b - (tempw * meanx) / stdx

print(f"equateion of line denormalized: {w}x + {b}")

#plotting points
plt.scatter(df['SquareFeet'],df['Price'])
plt.xlabel("Sq feet")
plt.ylabel("Price")

#plotting model


points = np.linspace(df['SquareFeet'].min(), df['SquareFeet'].max(), 400)
eq = w*points + b
plt.plot(points,eq,color='red')
plt.show()

