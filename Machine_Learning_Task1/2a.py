import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import torch

# Ucitavamo bottle fajl
df = pd.read_csv("data/bottle.csv")

# Biramo relevantne kolone i uklanjamo nedostajuce vrednosti
df = df[["Salnty", "T_degC"]].dropna()

# Koristimo prvih 700 uzoraka za trening
x_train = df["Salnty"].values[:700].reshape(-1, 1)  # Input za model
# primer X_train = [35.2, 34.8, 36.1, 33.7, 34.0] -> X_train = [[35.2], [34.8], [36.1], [33.7], [34.0]]
y_train = df["T_degC"].values[:700]                 # Target

# Normalizacija podataka, (originalnaVrednost - prosek) / standardnaDevijacija 
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std

# Pravljenje grafika
plt.figure(figsize=(12, 5))
plt.scatter(x_train, y_train, color='gray', label="Podaci")

# Priprema za racunanje greske
trosak = []                         
degrees = range(1, 7)               

# rucna implementacija polinomijalne regresije u torchu kao sto ste rekli u mejlu

def train_polynomial_regression(x, y, degree, epochs=2000, lr=0.001):
    # Transformacija ulaza u polinomijalne osobine
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)     # daje vise dimenzija input podacima

# npr. za stepen 2 pretvara 
# x_train = [[35.0], [34.8], [36.1]] ->   
# x_poly = [[1, 35.0, 1225.0],               jedinice su za bias tako da ide, [1,x,x^2]
#           [1, 34.8, 1211.04],
#           [1, 36.1, 1303.21]]
    
    # Konvertovanje u PyTorch tensore
    X = torch.tensor(x_poly, dtype=torch.float32)             # 32 bitni pytorch tensor
    Y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)   # ovaj mora 2D

# tensor oblik je npr za stepen 2, (700,3) zbog 1,x,x^2
    
    # Inicijalizacija parametara
    num_features = X.shape[1]    # shape nam vraca tuple tensora, uzimamo broj koliko featura ima
    W = torch.randn((num_features, 1))   # matrica tezine na osnovu featura, 1 je broj kolona. 
    # primer, za stepen 2 (3,1) -> [[w0], [w1], [w2]]
    b = torch.zeros(1)   # bias ovde, 1D sa jednim elementom, pocinjem sa nulom, shitfuje polinomijalnu krivu
    
    for epoch in range(epochs):
        # ovo je forward pass, predikcija tako sto mnoze matrice i dodaje bias
        # posto je X (700, num_features) a W (num_features,1) dimenzije tensora ce biti (700,1)
        Y_pred = X @ W + b # dakle gradimo predikcije matricnim mnozenjem inputa sa weightovima i biasom
        # Ypred = w0 * 1 + w1 * x + w2 * x^2 itd.
        
        # ovo je backward pass, funckija troska 
        # trosak L = 1/n (Ypred - Y)^2  treba da se minimizuje - prosecna kvadratna razlika izmedju predikcije i poznatih podataka

        # Y_pred - Y je greska
        # mnozi se sa 2 jer stepen prelazi ispred nakon parcijalnog izvoda
        grad_Y_pred = 2.0 * (Y_pred - Y) / Y.size(0)

        # X se transponuje da bude mnoziva
        # kad se pomnozi sa gradijentom predikcije dobija se koliko svaki stepen doprinosi gresci
        grad_W = X.T @ grad_Y_pred
        grad_b = torch.sum(grad_Y_pred, dim=0) # skalar, bias je samo jedna vrednost

        # rucno menjamo parametre tako da gasimo gradijent
        with torch.no_grad():
            W -= lr * grad_W # weightovi se pomeraju tako da se smanji greska (skalirano sa learning rateom)
            b -= lr * grad_b # isto i za bias
            
        
    Y_pred = X @ W + b  # konacna predikcija posle treninga
    final_loss = torch.mean((Y_pred - Y) ** 2).item()   # tensor u brojeve pomocu .item()
    return Y_pred.numpy(), final_loss

# Iteracija kroz stepene polinoma
for degree in degrees:
    # treniranje modela na normalizovanim podacima
    y_pred_norm, loss = train_polynomial_regression(x_train_norm, y_train_norm, degree)
    
    # denormalizacija predikcija za crtanje
    y_pred = y_pred_norm * y_std + y_mean
    
    # Prikaz regresione krive
    sorted_indices = np.argsort(x_train.ravel())  # prvo flatten u 1D, i onda uzima indekse pozicija kojim redom treba da budu
    x_sorted = x_train.ravel()[sorted_indices]    # sada ih sortiramo po tom redu
    y_sorted = y_pred[sorted_indices].ravel()     # a i ovde da bi bile uparene naravno
    
    plt.plot(x_sorted, y_sorted, label=f"Stepen {degree}")
    
    # cuvanje funkcije troska
    trosak.append(loss)
    print(f"Stepen {degree}, Trošak: {loss}")  

plt.xlabel("Salinitet")
plt.ylabel("Temperatura")
plt.legend()
plt.title("Polinomijalna regresija različitih stepena")
plt.savefig("2a_1.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(degrees, trosak, marker='o', linestyle='dashed', color='blue')
plt.xlabel("Stepen polinoma")
plt.ylabel("Funkcija troška")
plt.title("Funkcija troška u zavisnosti od stepena polinoma")
plt.savefig("2a_2.png")
plt.close()

# Sta sam primetio:
# Kada koristimo nize stepene polinoma (npr. stepen 1 ili 2), regresione krive ne uspevaju dobro da prate podatke, sto ukazuje na underfitting
# Vidimo da se funkcija troska (bar u proseku) smanjuje kako povecavamo stepen polinoma
# Tendencija je da ide na dole al nekad malo ume da mu ga da
# U visim stepenima ume da bude overfitovanje na grafiku 2a_1, malo krece da se raspada
