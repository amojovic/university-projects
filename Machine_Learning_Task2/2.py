import numpy as np
import pandas as pd
import json
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=16, output_size=1, learning_rate=0.01):
        # ovde pravim random weightove, množim sa 0.1 da bi bile male vrednosti, biasi su nula na početku kao uvek
        # želim jednostavnu strukturu input -> hidden -> output  u našem slučaju 7->16->1  
        # 1 neuron je dovoljan na kraju jer mi samo treba 0 ili 1 za survived 
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    # pošto moram da klasifikujem da li je lik crko ili nije tj. 1 ili 0 onda mi treba sigmoid
    # u suštini  bilo koji broj od -inf do +inf napakuje da bude između 0 i 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # ovo mu dođe formula
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)           # pošto moram da uradim backprop onda mi treba izvod funkcije
    
    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)  # standardna aktivaciona funckija X*W + b
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    
    def backward(self, X, y, output):
        error = y - output   # error = istina - predikcija
        d_output = error * self.sigmoid_derivative(output) # sada imam gradijent gubitka u odnosu na output kada sam ga skalirao
        
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden)
        
        # sada prvo nazad output -> hidden update
        self.weights2 += self.hidden.T.dot(d_output) * self.learning_rate
        self.bias2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        
        # a ovde sada ide hidden -> input
        self.weights1 += X.T.dot(d_hidden) * self.learning_rate
        self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
    

    # lagano treniranje forward + backward poziv
    def train(self, X, y, epochs=1000, print_loss_every=100):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            # Ispisujem gubitak na 100 epoha da bih proverio da li se smanjuje kako treba
            if epoch % print_loss_every == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoha {epoch}, Gubitak: {loss:.3f}")
    
    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        return (output > threshold).astype(int) # ako više od 50% onda preživi, True astype int je 1
    
    def save(self, path='model.json'):
        # volim da čuvam modele kao JSON jer su čitljivi u toj formi
        model_data = {
            'weights1': self.weights1.tolist(),
            'bias1': self.bias1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias2': self.bias2.tolist()
        }
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load(self, path='model.json'):
        with open(path, 'r') as f:
            model_data = json.load(f)
        self.weights1 = np.array(model_data['weights1'])
        self.bias1 = np.array(model_data['bias1'])
        self.weights2 = np.array(model_data['weights2'])
        self.bias2 = np.array(model_data['bias2'])

def pripremi(df):
    df = df.copy()
    
    # ovo isto kao prošli put
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # sada ću da normalizujem, moram da uzmem .values za numpy jer pandas ima autizam
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].values
    
    # ovako se normalizuje, poenta je dobiti manje vrednosti za brži trening
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X

izbor = input("Unesi 1 za treniranje novog modela ili 2 za učitavanje postojećeg modela: ").strip()
input_size = 7  # mislim zbog ovih 7 featura koje sam odabrao
nn = NeuralNetwork(input_size=input_size, hidden_size=16, learning_rate=0.01)

if izbor == '1':
    train_data = pd.read_csv('train.csv')
    X_train = pripremi(train_data)
    y_train = train_data['Survived'].values.reshape(-1, 1)   # opet ono standardno gde ga petvorimo u kolonu
    
    nn.train(X_train, y_train, epochs=1001, print_loss_every=100)
    nn.save()
    print("\nModel sačuvan kao 'model.json'")
    
elif izbor == '2':
    if not os.path.exists('model.json'):
        print("Nema modela")
        exit()
    nn.load()
    print("Model učitan'")
    
else:
    print("Ovo nije bila opcija brt.")
    exit()

test_data = pd.read_csv('test.csv')
X_test = pripremi(test_data)
y_test = test_data['Survived'].values.reshape(-1, 1)

# Tačnost za ovaj je oko 90% što deluje ok
predictions = nn.predict(X_test)
test_accuracy = np.mean(predictions == y_test)
print(f"Tačnost: {test_accuracy:.3f}")

with open('predikcije.txt', 'w') as f:
    f.write('PassengerId,Survived\n')
    for pid, pred in zip(test_data['PassengerId'], predictions):
        f.write(f'{pid},{pred[0]}\n')

print("Predikcije sačuvane kao 'predikcije.txt'")