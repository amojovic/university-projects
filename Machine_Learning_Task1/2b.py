import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import torch

df = pd.read_csv("data/bottle.csv")

df = df[["Salnty", "T_degC"]].dropna()

x_train = df["Salnty"].values[:700].reshape(-1, 1)
y_train = df["T_degC"].values[:700]

x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std


lambda_values = [0, 0.001, 0.01, 0.1, 1, 10, 100]


plt.figure(figsize=(12, 5))
plt.scatter(x_train, y_train, color='gray', label="Podaci")

trosak = []


def train_polynomial_regression_with_l2(x, y, degree=4, lambda_val=0, epochs=2000, lr=0.001):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)

    X = torch.tensor(x_poly, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    

    num_features = X.shape[1]
    W = torch.randn((num_features, 1), requires_grad=True) 
    b = torch.zeros(1, requires_grad=True)
    
    for epoch in range(epochs):
        Y_pred = X @ W + b

        grad_Y_pred = 2.0 * (Y_pred - Y) / Y.size(0)
        grad_Y_pred = 2.0 * (Y_pred - Y) / Y.size(0)

        grad_W = X.T @ grad_Y_pred
        grad_b = torch.sum(grad_Y_pred, dim=0)
        
        # L2 regularizacija: lambda * ||W||^2
        # izvod: d/dW lambda * W^2 = 2 * lambda * W
        with torch.no_grad():
            W -= lr * (grad_W + 2 * lambda_val * W)
            b -= lr * grad_b # bias nema potrebe da regularizujemo
    
    Y_pred = X @ W + b
    final_loss = torch.mean((Y_pred - Y) ** 2).item()  
    return Y_pred.detach().numpy(), final_loss


for lambda_val in lambda_values:
    y_pred_norm, loss = train_polynomial_regression_with_l2(x_train_norm, y_train_norm, degree=4, lambda_val=lambda_val)
    
    y_pred = y_pred_norm * y_std + y_mean
    
    sorted_indices = np.argsort(x_train.ravel())
    x_sorted = x_train.ravel()[sorted_indices]
    y_sorted = y_pred[sorted_indices].ravel()
    
    plt.plot(x_sorted, y_sorted, label=f"lambda = {lambda_val}")
    
    trosak.append(loss)
    print(f"Lambda {lambda_val}, Trošak: {loss}") 

plt.xlabel("Salinitet")
plt.ylabel("Temperatura")
plt.legend()
plt.title("Polinomijalna regresija stepena 4 sa L2 regularizacijom")
plt.savefig("2b_1.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(lambda_values, trosak, marker='o', linestyle='dashed', color='blue')
plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Funkcija troška")
plt.title("Funkcija troška u zavisnosti od lambda")
plt.savefig("2b_2.png")
plt.close()

# vecinski vazi sve sto i za prethodni, sem sto ima lambda i stepen je cetvrti
# osim sto lambda na velikim vrednostima malo samelje 2b_1
# na lambda = 100 se toliko regularizuje da zavisnost postaje linearna, sto bas i ne treba tako
# na 2b_2 se vidi da, sto je veca lambda, funkcija troska raste
# objasnjenje ovoga? pa, po formuli za funkciju troska:
# l2_loss = sum(y_real - y_pred)^2 + lambda * sum(||W^2||)
# tako da, ako se lambda povecava, funkcija troska ce isto da raste posto se sabira