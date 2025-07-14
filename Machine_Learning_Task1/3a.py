import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder    # da bi mogao da prebacim kategorije -> brojeve
import torch

data = pd.read_csv('data/iris.csv')

x = data[['sepal_length', 'sepal_width']].values  # prva dva feature-a

# Uzimanje ciljne promenljive (vrste cveta) kao izlaznih podataka
y = data['species'].values

le = LabelEncoder()

y = le.fit_transform(y)  # nazive Iris-setosa, Iris-versicolor, Iris-virginica u brojeve 0,1,2

# nasumicno uzimam podatke za split (ovo moram ovako jer su sortirani), recimo 70-30 odnos 
# stratify je da bi bili ravnomerno rasporedjeni po klasama
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

# Pretvaranje trening i test podataka u tensore za dalju obradu
x_train_torch = torch.tensor(x_train, dtype=torch.float32)
x_test_torch = torch.tensor(x_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# ideja knn je da gledamo "najblize komsije" nekog podatka pri pravljenju predikcije
# potrebna je nekakva definicija pojma "najblize", koristi se tipicno Euklidska ili Menhetn distanca
# koristicemo Euklidsku jer je based.

# bira se k, kolko komsija uzimamo
# za svaki novi podatak, racuna se distanca od njega do svakog drugog podatka u datasetu
# sortiraju se rezultati i bira se k najblizih

def knn(x_train, y_train, x_test, k=3):
    # broj test primera
    n_test = x_test.shape[0]
    
    # inicijalizacija tensora za predikcije
    predictions = torch.zeros(n_test, dtype=torch.long)

    for i in range(n_test):
        distances = torch.sqrt(((x_train - x_test[i]) ** 2).sum(dim=1))
        # primer kako radi:
        # imamo npr. x_train: 5 trening tačaka sa koordinatama [[1, 2], [2, 3], [3, 1], [4, 4], [5, 2]]
        # y_train: labele [0, 1, 0, 1, 2].
        # x_test[0] = [2, 2]
        # k = 3
        # prvo: x_train - x_test[0] = [[1-2, 2-2], [2-2, 3-2], [3-2, 1-2], [4-2, 4-2], [5-2, 2-2]] = [[-1, 0], [0, 1], [1, -1], [2, 2], [3, 0]]
        # kvadriranje: [[1, 0], [0, 1], [1, 1], [4, 4], [9, 0]]
        # sumiranje po dim=1: [1, 1, 2, 8, 9]
        # koren: distances = [1.0, 1.0, 1.41, 2.83, 3.0]

         
        # pronalazenje indeksa k najblizih suseda (najmanja rastojanja)
        # Uzimamo 3 najmanja: [1.0, 1.0, 1.41]
        # Indeksi: [0, 1, 2]
        _, indices = torch.topk(distances, k, largest=False)
        
        # uzimanje labela k najblizih suseda
        # y_train = [0, 1, 0, 1, 2]
        # indices = [0, 1, 2]
        # k_nearest_labels = [0, 1, 0]
        k_nearest_labels = y_train[indices]
        
        # vecinsko glasanje: pronalazak najcesce labele medju susedima
        # unique = [0, 1], counts = [2, 1] (0 se pojavljuje 2 puta, 1 jednom)
        unique, counts = torch.unique(k_nearest_labels, return_counts=True)
        predictions[i] = unique[torch.argmax(counts)]  # indeks najveceg broja glasova
    
    return predictions

# predikcija na test skupu koristeci k-NN sa k=3
y_pred_torch = knn(x_train_torch, y_train_torch, x_test_torch, k=3)

accuracy = (y_pred_torch == y_test_torch).float().mean().item()
# primer kako radi accuracy, ovo vraca torch bool niz u sustini:
# y_pred_torch = [0, 1, 2, 1, 0]
# y_test_torch = [0, 1, 1, 1, 2]
# y_pred_torch == y_test_torch = [True, True, False, True, False]

print(f'Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(8, 6))


unique_classes = np.unique(y_train)

for class_label in unique_classes:
    class_points = x_train[y_train == class_label] # filtriram x_train tako da imam samo one koje pripadaju odgovarajucoj klasi
    x_coords = class_points[:, 0]  # sve redove uzimam ali samo kolonu sepal_length
    y_coords = class_points[:, 1] 
    class_name = le.classes_[class_label]  # vraca nazad klasu u tekst 
    plt.scatter(x_coords, y_coords, label=class_name)

x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()

# ovo je poremeceno i defektno ali radi:
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)) # mreza 100 tacaka izmedju min i max 

# pretvaranje mreze u tensor za k-NN predikciju
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32) # moram da flattenujem podatke i kombinujem ih pomocu np.c, sada mozemo da koristimo model da predvidimo kojoj grupi pripada svaka tacka

# predikcija klasa za sve tačke u mrezi
z_torch = knn(x_train_torch, y_train_torch, grid_points, k=3)

z = z_torch.numpy().reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.3)  # crta obojene konture, alpha za providnost

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('k-NN Klasifikacija (k=3)')

plt.legend()

plt.savefig('3a.png')

plt.show()