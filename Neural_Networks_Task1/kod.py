import numpy as np
import nnfs
import pickle
import gzip
import os
import json  

nnfs.init()

# Load MNIST data
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation, Rectified Linear Unit (ne moze biti negativna)
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        # ovo gleda levo desno tj. sve argumente i bira veci, tkd. nikada nemamo nista negativno
        # ovo je poenta rektifikacije

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        # ja sam to bio spustio na 0 u forwardu, tako da sad ne saljem nista nazad

# logiti > softmax > verovatnoće
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)          # ovde ih softmaxuje
        self.output = self.activation.output     # ovo su verovatnoce
        return self.loss.calculate(self.output, y_true)  # ovo je loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:           
            y_true = np.argmax(y_true, axis=1)  # ako su one hot vraca ih u obican oblik
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1  # simulira se ovo tako sto se na pravom indeksu oduzme 1: dL/dz = y_pred - y_true
        self.dinputs = self.dinputs / samples      # inace bi morala cela jedna one hot matrica da se napravi

# softmax poenta je da se predstavi neuroni po normalnoj raspodeli tj. kolika je verovatnoca za koji izlaz
# formula je softmax = e^(vektor podatka) / suma_ svih(e^(vektor podataka))
# vektor podatak je ustvari kako bi neki sloj konkretno predstavljao taj podatak
# eksponenc omogucava da nema negativnih vrednosti
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # ovo je za skaliranje, exp(1000) bi bio overflow npr.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # primena formule (sve budu sada izmedju 0 i 1)
        self.output = probabilities
    
    # na primer, ako batch ima 3 slike, i 10 klasa, self.output je shape (3, 10) 3 output vektora po 10 verovatnova svaki.
    # matrica 3 reda 10 kolona u sustini
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)   # ovo pretvara niz u kolonu
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    # formule da bih razumeo kako se pravi Jakobijan: 
    # davlues je gradient loss-a u odnosu na softmax outpute tj. ∂L/∂s
    # Jij = dsi / dzj
    # kad i == j onda ds/dz = si * (1 - si)
    # kad i nije j onda - si*sj
    # J=diag(s)−s*s^T

# kapiram pomocna loss klasa za pozivanje specificnih forwarda, takodje ima ovo za sredji loss nekog batcha
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# poenta je uraditi loss funkciju, tj. da se izracuna gubitak
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)  # ovo je broj slika
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # ogranici za obe strane da ne bude log(0)
        if len(y_true.shape) == 1:  # ako nije one hot i imamo 3 slike npr. onda je samo niz labela [4,2,9]
            correct_confidences = y_pred_clipped[range(samples), y_true]  # verovatnoce za tacne slike
        elif len(y_true.shape) == 2: # ako jeste one hot onda je matrica gde umesto svake labele ima niz [[0,0,0,1,0...],...]
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # posto su dve matrice istigh dimeznija sada moze ovako
        negative_log_likelihoods = -np.log(correct_confidences) # ovo je po formuli iz teorije
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]   # ovo ga pretvara da bude one hot 
        self.dinputs = -y_true / dvalues      # formula je dL/dVal = -y/dVal / N gde su y istinite vrednosti a dVal softmax output
        self.dinputs = self.dinputs / samples # formula treba da se dobije izvodom ovoga L = -suma(y * log(Val)) iz forwarda

# Adam optimizer
# ok znaci skraceno za Adaptive Moment Estimation
# koristi:
# prosecan gradijent (momentum)
# prosecan kvadrat gradijenta (RMSprop)
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay    # koliko brzo se smanjuje learning rate
        self.iterations = 0
        self.epsilon = epsilon  # sitna vrednost da se ne deli sa nulom
        self.beta_1 = beta_1 # momentum faktor
        self.beta_2 = beta_2 # RMSprop faktor

    # ovo smanjuje learn rate kroz epohe
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):  # prima sloj koji ima trenutne parametre i gradijente
        if not hasattr(layer, 'weight_cache'):   # samo inicijalizacija memorije ako je vec nema
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # ideja je da gradijent dweight i dbias nam kaze u kom smeru treba da idemo ali necemo da skocimo ka njima odma
        # nego koristimo prethodni pravac tj. momentum, znaci ova formula m = beta * m + (1-beta) * gradijent

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # ovo su korekcije, momentumi se pojacaju tako sto ih delimo sa 1-beta^(iteracija+1)

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # ovo je ideja iz RMSpropa, ako neki gradijent stalno varira onda je nestabilan
        # cache je kao istorija kvadrata gradijenta za svaki parametar, u sustini pomocne promenljive
        # formula(samo je sada beta 2): cache = beta * cache + (1-beta) * gradijent^2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        # ista korekcija kao za momentume

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        # ovde se konacno menjaju tezine
        # formula je: W = -lr * m/(koren(cache) + epsilon) 

    def post_update_params(self):
        self.iterations += 1    # ovo je jasno

# shape je ovo: (batch_size, channels, height, width)
# flattenuje 28*28 = 784, zadrzava batch size, ostalo inferuje
class Layer_Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)
        
    # gradijenti su derivati iz sledeceg sloja, hocemo da ih vratimo u 2D
    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)

# Neural network class
class NeuralNetwork:
    def __init__(self):
        self.flatten = Layer_Flatten()
        self.dense1 = Layer_Dense(784, 128)  # 28*28 = 784 input featura(tj. pikseli), 128 neurona
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(128, 64)   # sada 128 featura iz proslog sloja -> sledeci od 64 neurona
        self.activation2 = Activation_ReLU()
        self.dense3 = Layer_Dense(64, 10)    # konacno na 10 klasa, cifre od (0-9) 
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        # ovde sam mogao da menjam broj slojeva (naravno mora jos izmena dole)
        # sto je veci broj i slojeva i neurona u sloju to treniranje traje duze
        # tako da sam ostao na kraju sa 2 obicna sloja i 1 izlazni
        
    def forward(self, X, y_true=None):  # x je batch slika [batch_size,28,28]
        self.flatten.forward(X)         # prebacim u 784
        self.dense1.forward(self.flatten.output)
        self.activation1.forward(self.dense1.output)   # 1 sloj + ReLu znaci sada [batch_size, 64]
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)   # 2 sloj + ReLu posle ovoga je [batch_size, 10]
        self.dense3.forward(self.activation2.output)
        if y_true is not None:
            return self.loss_activation.forward(self.dense3.output, y_true)  # ako imamo labele, tj. kod treniranja vracamo loss
        return self.loss_activation.activation.output  # ako nema onda verovatnoce
    
    def backward(self, dvalues, y):            # ovo je sada za treniranje mreze 
        self.loss_activation.backward(dvalues, y)            
        self.dense3.backward(self.loss_activation.dinputs)   # propagiranje unazad kroz sve slojeve 
        self.activation2.backward(self.dense3.dinputs)       # svaki sloj izracuna svoj gradijent i prosledi prethodnom 
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)
        self.flatten.backward(self.dense1.dinputs)

    def save(self, filename):                        # za cuvanje i ucitavanja svih parametra iz modela u JSON formatu
        model_data = {
            'weights1': self.dense1.weights.tolist(),  # Konvertujemo NumPy niz u listu
            'biases1': self.dense1.biases.tolist(),
            'weights2': self.dense2.weights.tolist(),
            'biases2': self.dense2.biases.tolist(),
            'weights3': self.dense3.weights.tolist(),
            'biases3': self.dense3.biases.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)  # Cuvamo kao JSON

    def load(self, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
            self.dense1.weights = np.array(model_data['weights1'], dtype=np.float32)  # Konvertujem listu nazad u NumPy niz
            self.dense1.biases = np.array(model_data['biases1'], dtype=np.float32)
            self.dense2.weights = np.array(model_data['weights2'], dtype=np.float32)
            self.dense2.biases = np.array(model_data['biases2'], dtype=np.float32)
            self.dense3.weights = np.array(model_data['weights3'], dtype=np.float32)
            self.dense3.biases = np.array(model_data['biases3'], dtype=np.float32)

# Podela podataka na grupe
training_data, validation_data, test_data = load_data()
X_train, y_train = training_data
X_val, y_val = validation_data
X_test, y_test = test_data

# Matrica 28 x 28 za slike, 784 vrednosti inace
X_train = X_train.reshape(-1, 28, 28)
X_val = X_val.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# sprema ove objekte
model = NeuralNetwork()
optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-3)
model_filename = 'model.json'  # Promenjeno na .json za JSON format

# Prvo proveravam da li korisnik zeli da ucita sacuvani model
load_model = input("Da li želite da učitate sačuvani model? (da/ne): ").strip().lower() == 'da'
if load_model and os.path.exists(model_filename):
    print(f"Učitavanje modela iz {model_filename}...")
    model.load(model_filename)
    # Testiranje ucitanog modela
    test_loss = model.forward(X_test, y_test)
    test_predictions = np.argmax(model.loss_activation.output, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f'\nRezultati učitanog modela na test skupu:')
    print(f'Tacnost: {test_accuracy:.3f}')
    print(f'Loss: {test_loss:.3f}')
else:
    # Ako nema ucitavanja, treniram novi model
    print("Treniranje novog modela...")
    for epoch in range(1001):
        # Trening forward prolaz
        train_loss = model.forward(X_train, y_train)
        train_predictions = np.argmax(model.loss_activation.output, axis=1)  # uzme labelu za najvecom verovatnocom
        train_accuracy = np.mean(train_predictions == y_train)  # prosecna tacnost za jedan batch
        # npr. ako su predictions = np.array([2, 0, 4, 1, 3]) i y_train = np.array([2, 1, 4, 1, 0])
        # onda bi dobili [True,False,True,True,False] a mean tretira to kao [1,0,1,1,0]
        # sto je onda 1+0+1+1+0/5 = 0.6 ili 60% tacnosti
        train_output = model.loss_activation.output.copy()  # Cuvam trening izlaz za backpropagation

        # Odmah radim backpropagation, dok su dimenzije sigurne
        model.backward(train_output, y_train)
        optimizer.pre_update_params()
        optimizer.update_params(model.dense1)
        optimizer.update_params(model.dense2)
        optimizer.update_params(model.dense3)
        optimizer.post_update_params()

        # Validacija na svakih 100 epoha, koristim X_val i y_val, ne koristim je za nesto specijalno kao regulisanje overfittinga
        # samo za ispis i pracenje stanja
        if epoch % 100 == 0:
            val_loss = model.forward(X_val, y_val)
            val_predictions = np.argmax(model.loss_activation.output, axis=1)
            val_accuracy = np.mean(val_predictions == y_val)

            # ispis na svakih 100 epoha, dodajem validacione metrike
            print(f'Epoha: {epoch}, '
                  f'Trening Tacnost: {train_accuracy:.3f}, '
                  f'Trening Loss: {train_loss:.3f}, '
                  f'Validacija Tacnost: {val_accuracy:.3f}, '
                  f'Validacija Loss: {val_loss:.3f}, '
                  f'Learning Rate: {optimizer.current_learning_rate:.6f}')


    # Testiranje treniranog modela
    test_loss = model.forward(X_test, y_test)
    test_predictions = np.argmax(model.loss_activation.output, axis=1)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f'\nRezultati:')
    print(f'Tacnost: {test_accuracy:.3f}')
    print(f'Loss: {test_loss:.3f}')

    # Cuvanje modela
    print(f"Čuvam model kao {model_filename}")
    model.save(model_filename)