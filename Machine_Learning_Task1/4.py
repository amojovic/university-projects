import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv("data/disaster-tweets.csv")

# ove reci zelimo da poizbacujemo u preprocesuiranju teksta
# ako se to ne uradi, zagadjivace rezultate
# mrzelo me da skidam recnik.
stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
                  'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                  'through', 'during', 'before', 'amp', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                  'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                  'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
                  'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                  'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 
                  'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 
                  'wasn', 'weren', 'won', 'wouldn', 'via'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # izbacujemo linkove
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # izbacujemo tagovanje ljudi (@neko_nesto)
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # izbacujemo #current_thing
    text = re.sub(r'[^a-zA-Z\s]', '', text) # izbacujemo sve sem obicnog teksta; znakovi interpunkcije, brojevi, itd lete
    text = ' '.join([word for word in text.split() if word not in stop_words]) # i onda stop words
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
x = df['cleaned_text'].values
y = df['target'].values

def build_vocabulary(texts, n=10000):
    word_counts = Counter(' '.join(texts).split()) # broji se pojavljivanje svake reci
    return [word for word, _ in word_counts.most_common(n)] # vraca se n najcescih reci

# bag of words: vraca np matricu gde je svaki red tekst, a svaka kolona rec iz recnika
# za texts = ["pojma nemam", "samo lupam jer nemam inspiraciju"]
# i vocab = ["pojma", "nemam", "samo", "lupam", "jer", "inspiraciju"]
# ova funkcija vraca:
# [[1, 1, 0, 0, 0, 0],
#  [0, 1, 1, 1, 1, 1]]
def texts_to_bow(texts, vocab):
    bow = np.zeros((len(texts), len(vocab)))
    # enumerate u svakoj iteraciji daje indeks i objekat; malo namazanija klasicna for petlja
    # idemo kroz svaki tekst
    for i, text in enumerate(texts):
        # i onda idemo kroz svaku rec u datom tekstu
        for word in text.split():
            if word in vocab:
                bow[i, vocab.index(word)] = 1 # self-explanatory poprilicno
    return bow

# training
# prosledjuje se konacni bag of words zajedno sa klasama za svaki tekst (disaster/not disaster)
# alpha nam treba za Laplasov smoothing da se izbegne verovatnoca nula
def fit_naive_bayes(x_bow, y, vocab_size, alpha=1.0):
    n_samples = len(y)
    classes = np.unique(y) # niz klasa (u sustini samo 0 i 1)
    n_classes = len(classes) # kolko ih ima
    priors = np.zeros(n_classes) # P(class), verovatnoca svake klase
    word_probs = np.zeros((n_classes, vocab_size)) # uslovna verovatnoca pojavljivanja reci po klasi, tj P(word | class)
    
    for idx, c in enumerate(classes): # idemo po svakom dokumentu koji pripada klasi c
        x_c = x_bow[y == c] # uzimamo sve dokumente klase c
        priors[idx] = x_c.shape[0] / n_samples # gradi se verovatnoca klase

        # po svim dokumentima klase c brojimo pojavljivanja reci
        # ovo daje ukupan broj pojavljivanja svake reci u toj klasi
        word_counts = x_c.sum(axis=0) + alpha # i naravno radi se laplas

        word_probs[idx] = word_counts / word_counts.sum() # uslovna verovatnoca
    
    return priors, word_probs, classes

# sa rezultatima od treniranja se sad radi predikcija
def predict_naive_bayes(x_bow, priors, word_probs, classes):
    # uzima se logaritam od a priori verovatnoca
    # word_probs.T je transponovana matrica oblika (vocab_size, n_classes)
    # kad se to pomnozi sa x_bow, dobija se logaritamska verovatnoca dokumenta za svaku klasu
    # tako da se dobija na kraju po jedan vektor po klasi
    # kad se saberu, dobija se krajnja verovatnoca
    log_probs = np.log(priors) + (x_bow @ np.log(word_probs.T))

    # i vraca se najverovatnija klasa
    # np.argmax(...) daje, za svaki dokument, indeks klase sa najvisom verovatnocom
    # i onda se mapira nazad na klasu i vraca se
    return classes[np.argmax(log_probs, axis=1)]

    # zasto logaritmi?
    # pa, da se radi sa linearnim vrednostima, mnozilo bi se puno malih brojeva, sto bi patilo od fp rounding errors
    # samo po sebi tu bi se gubile informacije, a pogotovo ako dodje do underflowa
    # ovako je prosto stabilnije, a sustina ostaje ista

# sad idemo triput Srpski kroz dataset
# pravimo bag of words od teksta
# treniramo
# radimo predikciju
# printujemo
# i cuvamo dobijen accuracy
accuracies = []
for i in range(1, 4):
    # x su tekstovi, y klase
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80% training, 20% test
    vocab = build_vocabulary(x_train) # pravi se recnik od training podataka
    x_train_bow = texts_to_bow(x_train, vocab) 
    x_test_bow = texts_to_bow(x_test, vocab)
    
    priors, word_probs, classes = fit_naive_bayes(x_train_bow, y_train, len(vocab))
    y_pred = predict_naive_bayes(x_test_bow, priors, word_probs, classes) # i onda radimo predikciju na test samplu
    
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy {i}: {accuracy * 100:.2f}%')
    accuracies.append(accuracy)

print(f'Prosečan Accuracy: {np.mean(accuracies) * 100:.2f}%')

# 0 - disaster, 1 - normalno
positive_tweets = df[df['target'] == 1]['cleaned_text']
negative_tweets = df[df['target'] == 0]['cleaned_text']
positive_words = ' '.join(positive_tweets).split()
negative_words = ' '.join(negative_tweets).split()
positive_word_count = Counter(positive_words)
negative_word_count = Counter(negative_words)

# sve manje vise self explanatory
print("\n5 najčešćih reči u pozitivnim tvitovima:")
print(positive_word_count.most_common(5))
print("\n5 najčešćih reči u negativnim tvitovima:")
print(negative_word_count.most_common(5))

# izvlaci se likelihood ratio sad
# samo rec koja se pojavljuje makar 10 puta u obe klase ulazi u opticaj
def calculate_lr(word):
    if positive_word_count[word] >= 10 and negative_word_count[word] >= 10:
        return positive_word_count[word] / negative_word_count[word]
    return None

words_lr = {word: calculate_lr(word) for word in set(positive_word_count) & set(negative_word_count)}
# filtriramo, testiranje pokazuje da ovo prosto mora da se uradi
filtered_words_lr = {k: v for k, v in words_lr.items() if v is not None}
# onda ih sortiramo
sorted_words_lr = sorted(filtered_words_lr.items(), key=lambda x: x[1], reverse=True)

# sad su "najpozitivnije" reci na pocetku niza, a "najnegativnije" na kraju
# tako da se printuju levih i desnih 5 komada

print("\n5 reči sa najvećim LR:")
print(sorted_words_lr[:5])
print("\n5 reči sa najmanjim LR:")
print(sorted_words_lr[-5:])

# komentari ---
# - negativne reci ('fire', 'disaster', ...) se pojavljuju u kontekstu katastrofa
# - pozitivne/neutralne (npr 'like', 'new') su u opstoj konverzaciji
# - LR metrika daje reci specificne za negativne (visok LR, npr. 'train') ili neutralne (nizak LR, npr. 'like') tvitove
# rezultati su smisleni