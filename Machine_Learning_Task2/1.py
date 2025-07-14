import pandas as pd
import numpy as np
import json
import os

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None           # ovde ću da čuvam strukturu stabla

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)  # ovo će da vrati tu sturkturu

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1:                          # ako ima samo jedna klasa onda svi pripadaju njoj
            return {'label': int(np.unique(y)[0])}          # znači pravim leaf node,tj. dictionary sa tim brojem
        if depth == self.max_depth or len(X) == 0:          # ako ili nema više ljudi ili smo stigli do max dubine onda
            return {'label': int(np.bincount(y).argmax())}  # uzme najzastupljeniju klasu

        # interesantno kako radi ovaj bincount, npr. za y = [2, 2, 2, 3, 3] bi bilo [0,0,3,2] 
        # jer se 0 i 1 ne pojavaljuju ni jednom i onda argmax moze da vrati korektan broj

        best_split = self.best_split(X, y)
        if best_split is None:
            return {'label': int(np.bincount(y).argmax())}   # ako ne može

        # rekurzivno dalje
        left_tree = self.build_tree(X[best_split['left']], y[best_split['left']], depth + 1)
        right_tree = self.build_tree(X[best_split['right']], y[best_split['right']], depth + 1)

        return {
            'feature': int(best_split['feature']),
            'threshold': float(best_split['threshold']),
            'left': left_tree,
            'right': right_tree
        }
        # ovo je čvor odluke i tu čuvamo feature na kom smo splitovali kao i threshold (tj granica) 
        # tako da ako feature <= threshold onda left

    def best_split(self, X, y):
        best_gini = float('inf')  # manji gini indeks je bolji tako da krećemo od beskonačnosti
        best_split = None
        n_samples, n_features = X.shape

        for feature in range(n_features):            # idem po svakoj osobini
            thresholds = np.unique(X[:, feature])    # jedistvene vrednosti menju ljudima za neku osobinu
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold  # pravim maske [True,False,True,False] npr. za svaki primer
                right_mask = X[:, feature] > threshold
                left_y = y[left_mask]                  # primenim masku tako da dobijem labelove 0 ili 1 za one koji idu levo
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:  # ako prazni onda sledeći threshold
                    continue

                gini = self.gini_index(left_y, right_y)   # poenta je naći feature i threshold takav da gini indeks bude najmanji i onda njih koristimo
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left': left_mask,
                        'right': right_mask
                    }

        return best_split


    # našli smo da može gini, entropija ili čak MSE, ali gini je najlakši i najbolji za klasifikaciju tako da smo njega odabrali
    def gini_index(self, left_y, right_y):  # ovo su oni labelovi za levu i desnu grupu posle splita
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        def gini(y):
            return 1 - sum((np.sum(y == c) / len(y)) ** 2 for c in np.unique(y))
            # ovo je na osnovu gini formule, znači 1 - suma(koliko se javlja neka klasa / ukupan broj) na kvadrat

        return (left_size / total_size) * gini(left_y) + (right_size / total_size) * gini(right_y)
        # pošto mi gini samo govori koliko je nečista neka grupa (tj. ako su mešani labelovi 1 i 0 onda nije čista)
        # y = [1, 1, 1, 1, 1]
        # Gini=1−(1)^2=1−1=0   ovo bi bilo savršeno čisto npr.
        # onda mi je bitno to što leva i desna grupa nisu neophodno iste veličine
        # zato radim weighted gini i to vraćam nazad


    def predict(self, X):
        return [self.predict_row(row, self.tree) for row in X]  # predviđanja za svakog individualno

    def predict_row(self, row, tree):
        if 'label' in tree:
            return tree['label']                        # ako je leaf vrati klasu
        if row[tree['feature']] <= tree['threshold']:   # ovo je ona provera da li idemo levo ili desno na osnovu granice
            return self.predict_row(row, tree['left'])
        else:
            return self.predict_row(row, tree['right'])

    def save(self, path='model.json'):
        with open(path, 'w') as f:
            json.dump(self.tree, f, indent=2)

    def load(self, path='model.json'):
        with open(path, 'r') as f:
            self.tree = json.load(f)


def priprema(df):
    df = df.copy()  # sprečava spam nekih warninga u konzoli

    df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].mean())       # prazne vrednosti u koloni menjam sa prosečnom
    df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # mode mi vraća najčešće mesto odakle su ljudi krenuli, tako da to umesto NaN

    df.loc[:, 'Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # olakšavam dalje da male bude 0 a female 1 tokom učenja
    df.loc[:, 'Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}) # isto

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[features]


model = DecisionTree(max_depth=5)  # ovo mi daje prihvatljivu tačnost 
izbor = input("Unesi 1 za treniranje novog modela ili 2 za učitavanje postojećeg modela: ").strip()

if izbor == '1':
    train_data = pd.read_csv('train.csv')
    X_train = priprema(train_data)
    y_train = train_data['Survived']

    model.fit(X_train.to_numpy(), y_train.to_numpy())
    model.save()
    print("Model sačuvan kao 'model.json'.")

elif izbor == '2':  # Ovde sam samo omogućio da može da se iskoristi sačuvan model
    if not os.path.exists('model.json'):
        print("Nema modela")
        exit()
    model.load()
    print("Model učitan")

else:
    print("Ovo nije bila opcija brt.")
    exit()

# Učitaj test skup i napravi predikciju
test_data = pd.read_csv('test.csv')
X_test = priprema(test_data)
y_test = test_data['Survived']  
test_predictions = model.predict(X_test.to_numpy())
test_data['Survived'] = test_predictions

# Tačnost je oko 95% za dubinu 5 štp deluje ok
test_accuracy = np.mean(test_predictions == y_test.to_numpy())
print(f"Tačnost na test skupu: {test_accuracy:.3f}")

# kad već mora da se čuva ispis u txt po specifikaciji
with open('predikcije.txt', 'w') as f:
    f.write('PassengerId,Survived\n')
    for pid, pred in zip(test_data['PassengerId'], test_predictions):
        f.write(f'{pid},{pred}\n')

print("Predikcije sačuvane kao 'predikcije.txt'.")
