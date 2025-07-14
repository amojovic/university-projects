import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # glupi tensorflow ima neke logove za info i warning, ostavljam samo errore da pokazuje

import torch                                                 # core stvari, biblioteke za tensore
import torch.nn as nn                                        # odavde dobijam konvolucione slojeve, linear itd.
import torch.optim as optim                                  # optimizeri Adam, SGD i ostali
import torchvision                                           # odavde mogu da pristupim ovom FashionMNIST datasetu
import torchvision.transforms as transforms                  # za prebacivanje slika u tensore i nomralizaciju
from torch.utils.data import random_split, DataLoader        # da razdeli train,val,test / loader za batcheve i shuffle

import tensorflow as tf                                      # glavna biblioteka
from tensorflow.keras import layers, models                  # keras ima Conv2D, Dense i Flatten slojeve
from tensorflow.keras.datasets import fashion_mnist          # fashion dataset

import numpy as np

def run_pytorch_model():
    print("\nPrvo PyTorch CNN na FashionMNIST:")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
# ovaj Compose je wrapper klasa, u suštini omogućava da se napakuje više transformacija koje se redom izvršavaju
# ToTensor ove PIL(python imaging library-pillow) slike pretvara u tensore (C,H,W) = (channel, height, width) npr. (1,28,28)
# ToTensor takođe napakuje da vrednosti piskela budu između 0 i 1. Pošto je ovo grayscale slike su 0-255. 
# normalizacija formula je output = (input - mean)/std ideja je da novi raspon bude [−1.0,1.0]
# ovako lakše konvergira, npr. -1 do 0 je crna, 0 do 0.5 siva a recimo nadalje bela
# ovo 0.5, je samo prvi od tupla, jer je ovo grayscale pa samo jedan kanal

    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# ovo je jasno, samo učita dataset za train i test, onda primeni transform

    train_len = int(0.8 * len(dataset))   # 80% je train, ovo je 80% od 60000 tako da 48000
    val_len = len(dataset) - train_len    # ostalo je validation, ovo je 12000
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len]) # random split slika

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
# u batcheve za efikasnost, shuffle na trening setu da ne nauci redosled napamet

    class CNN(nn.Module):     # pravim klasu koja nasledjuje od standardne klase nn.Module
        def __init__(self):
            super(CNN, self).__init__()                       # ovo hvata konstruktor iz tog modula
            self.conv_layers = nn.Sequential(                 # sada idem za redosled slojeva 
                nn.Conv2d(1, 32, kernel_size=3, padding=1),   
                # broj kanala 1(jer grayscale), 32 filtera 3x3 dimenzija za razlicite feature, u glavnom ivice. 
                # Padding od 1 zato sto filter od 3x3 centriran na prvom piskelu i viri van slike gore i levo za 1 piksel
                nn.ReLU(),       # rektifikacija vec radili, negativne vrednosti u nulu
                nn.MaxPool2d(2), # uzima blokove piksela 2x2 kao koraka i redukuje je sa 28x28 na 14x14, max u smilsu da ocuva najbitnije signale to jest najaci feature
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc_layers = nn.Sequential(   # sada ide fully connected deo
                nn.Flatten(),                 # posldenji layer je bio 64 batcha 7x7 nakon maxpoola
                nn.Linear(64 * 7 * 7, 128),   # 128 neurona
                nn.ReLU(),
                nn.Linear(128, 10)            # 10 outputa za svaku kategoriju u fashion mnistu
            )

        def forward(self, x):
            x = self.conv_layers(x)   # jasno
            x = self.fc_layers(x)
            return x

    model = CNN()
    criterion = nn.CrossEntropyLoss()  # za gubitak softmax za verovatnoce kategorija + negative log likelihood
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # vec radili, uzima u obzir momentum, menja lr na osnovu decay

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[PyTorch] Epoha {epoch+1}, Gubitak: {running_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():   # naravno ne trebaju mi grdijenti tokom evaluacije, to su na izvodi za backprop tako da sada to iskljucim
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # pogadja kategoriju
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[PyTorch] Tačnost: {100 * correct / total:.2f}%")


def run_tensorflow_model():
    print("\nDrugo TensorFlow CNN za FashionMNIST...")

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # učitam slike za trening i test

    x_train = x_train.astype("float32") / 255.0  # skaliram ih na da budu 0-1 jer grayscale
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1) # ovde je redosled (28,28,1) za 1 grayscale kanal
    x_test = np.expand_dims(x_test, -1)


    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # isti konv sloj kao za torch
        layers.MaxPooling2D((2, 2)),                                  # isto
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # same ista fora za padding
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),               # isto
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',                         # isto adam
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1) # ovde ima i verbose za progress bar, za torch bi nam trebala tqdm biblioteka

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[TensorFlow] Tačnost: {test_acc * 100:.2f}%")

run_pytorch_model()
run_tensorflow_model()

# Zaključak što se tiče poredjenja dva pristupa:
# tensorflow sa kerasom je znatno laksi za pisanje sto se tice sintakse
# ideja je bila da generalno budu isti, 2 konv sloja, 2 maxpoola i relu
# Konačne tačnosti su bile slične, praktično identične što je očekivano jer matematika iza svega ovoga u pozadini je isti i runnuje na istom sistemu
# Brzina izvršavanja zavisi od hardvera, koliko razumem TensorFlow je malo bolji na TPU ali na mom setupu je brzina izvršavanja bila ista.
# [PyTorch] Tačnost: 91.70%
# [TensorFlow] Tačnost: 91.80%
