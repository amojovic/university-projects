import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

data = pd.read_csv('data/iris.csv')
x = data[['sepal_length', 'sepal_width']].values  
y = data['species'].values

le = LabelEncoder()
y = le.fit_transform(y)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

x_train_torch = torch.tensor(x_train, dtype=torch.float32)
x_test_torch = torch.tensor(x_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

def knn(x_train, y_train, x_test, k):
    n_test = x_test.shape[0]
    predictions = torch.zeros(n_test, dtype=torch.long)
    
    for i in range(n_test):
        distances = torch.sqrt(((x_train - x_test[i]) ** 2).sum(dim=1))
        
        _, indices = torch.topk(distances, k, largest=False)

        k_nearest_labels = y_train[indices]
        unique, counts = torch.unique(k_nearest_labels, return_counts=True)
        predictions[i] = unique[torch.argmax(counts)]
    
    return predictions

accuracies = []
for k in range(1, 16):
    y_pred_torch = knn(x_train_torch, y_train_torch, x_test_torch, k)
    accuracy = (y_pred_torch == y_test_torch).float().mean().item()
    accuracies.append(accuracy)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 16), accuracies, marker='o', linestyle='-', color='blue')
plt.xlabel('Vrednost k')
plt.ylabel('Accuracy')
plt.title('Zavisnost tačnosti na test skupu u odnosu na k')
plt.xticks(range(1, 16))
plt.grid(True)
plt.savefig('3b.png')

best_k = np.argmax(accuracies) + 1  # +1 zbog toga sto range pocinje od 1
print(f'Najbolji k je {best_k} sa accuracy: {accuracies[best_k - 1] * 100:.2f}%')
plt.show()

# najbolje k svakako varira. na osnovu pokretanja vise puta, cesto ce biti izmedju 6 i 9 sa 75-85% accuracy
