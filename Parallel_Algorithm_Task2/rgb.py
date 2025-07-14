from PIL import Image          # za rad sa slikama
import numpy as np             # za rad sa nizovima i matricama
from functools import reduce   # reduce da se primeni funckija na sve elemente bez iteracija
import pickle
import os

BROJ_BINOVA = 8    # bin = u sustini sirina jednog bara na histogramu, imamo ih 8
class_names = []

def racunaj_histogram_1(image_path):
    image = Image.open(image_path).convert("RGB")     # RGB mod znaci da ima tacno 3 kanala za processing
    image_array = np.array(image)                     # 3D array(visina, sirina, broj_boja) sto ce biti 3 jer (r,g,b)
    bins = np.linspace(0, 256, BROJ_BINOVA + 1, endpoint=True)
    # vraca nam jednake raspone: array([  0. ,  32. ,  64. ,  96. , 128. , 160. , 192. , 224. , 256. ])
    # bin + 1 zato sto je nama u sustini bin prostor izmedju dve vrednosti
    # endpoints true da bi uzeli u obzir i 255
    # sam histogram predstavlja koliko ima piksela odredjenog intenziteta neke boje u nekoj slici
    # jedan histogram za svaku boju, ali vracamo sva 3 kao matricu
    
    num_pixels = reduce(lambda x, y: x * y, image_array.shape[:2])
    # vadimo samo height and width i mnozimo ih, reduce iz zezanja(nije potreban bio ovde)
    histograms = map(
        lambda ch: np.histogram(image_array[:, :, ch].ravel(), bins=bins, density=False)[0] / num_pixels,
        range(3)  # 0=Red, 1=Green, 2=Blue
    )
    # map ce da primeni lambda funkciju na svakom elementu
    # lambda ch uzima argument koja boja kanala je u pitanju
    # image_array[:, :, ch] uzima sve piksele odredjene boje
    # ravel pretvara 2D u 1D, znaci samo niz piksela odredjene boje
    # np.histogram racuna sam histogram
    # on je tuple 2 elementa [broj piksela, bin raspon] nama treba za normalizaciju samo broj piskela zato [0]
    # delimo broj pixela iz bin-a sa ukupnim brojem pixela zbog normalizacije
    # density=True bi nam dalo verovatnocu da pixel bude u bin-u dok nam deljenje daje deo pixela u tom binu
    
    # Vraca 3 reda, jedan za svaki kanal
    return np.array(list(histograms))

def racunaj_histogram_2(image_array):
    bins = np.linspace(0, 256, BROJ_BINOVA + 1, endpoint=True)
    num_pixels = reduce(lambda x, y: x * y, image_array.shape[:2])

    histograms = map(
        lambda ch: np.histogram(image_array[:, :, ch].ravel(), bins=bins, density=False)[0] / num_pixels,
        range(3)
    )
    return np.array(list(histograms))
    # iskoristio sam istu funckiju kao za jednu sliku samo joj sada prosledjujem array odmah a ne path

def moj_len(nekaLista):
    return reduce(lambda x, y: x + 1, nekaLista, 0)
    # posto nije dozvoljeno da se koristi len, pravim svoj len
    # reduce primenjuje ovu lambda fukciju koja koristi x da izbroji koliko elemanata
    # x je inicijalno naravno = 0


def prosecni_histogram(klasa_slike, parovi):
    
    class_images = list(map(lambda par: par[1], filter(lambda par: par[0] == klasa_slike, parovi)))
    histograms = list(map(racunaj_histogram_2, class_images))
    total_histogram = reduce(lambda x, y: x + y, histograms)
    
    return total_histogram / moj_len(class_images)

    # klasa 0 je npr. airplane, klasa 1 je automobile
    # parovi su npr. [(0, "path/to/image1.jpg"), (1, "path/to/image2.jpg"), (0, "path/to/image3.jpg")]
    # ako je klasa_slike 0 nama treba samo image1.jpg i image3.jpg
    # tako da filtriramo i od toga pravimo novu listu
    
    # histograms = list(map(calculate_histogram, class_images)) primenjuje funkciju na svakom clanu liste
    # total_histogram se dobija kao zbir svih i onda se radi prosek koji se vraca
    
    # npr. ako je:
    # calculate_histogram("img1.jpg") = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    # calculate_histogram("img3.jpg") = [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]
    # onda histograms ce biti [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]]
    # total_histogram je onda ovaj zbir: total_histogram = [[0.3, 0.5], [0.7, 0.9], [1.1, 1.3]]


def load_class_names(cifar_folder):
    global class_names  
    meta_file = os.path.join(cifar_folder, "batches.meta")
    with open(meta_file, 'rb') as f:
        meta_data = pickle.load(f, encoding='bytes')
        class_names = list(map(lambda name: name.decode('utf-8'), meta_data[b'label_names']))
    return class_names

# lista za nazive svih klasa

def ucitaj_cifar(cifar_folder, output_file):
    
    def load_batch(file_index):
        file_path = os.path.join(cifar_folder, f"data_batch_{file_index}")
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
            return images, labels
    # batch data je 2D array, 10 000 redova svaki je slika, 3027 kolone 1024 za svaku od RGB boja
    # svaka vrednost od 0-255
    # slike su 32x32 = 1024 piksela, ali x3 za svaku boju
    # batch lables je samo lista od 10 000 koja za svaku sliku ima kategoriju (airplane,automobile)
    # radimo reshape tako da iz (10 000, 3072) bude (10 000, 32, 32, 3) posle transpose-a
    # -1 sam preracunava da je stvarno 10 000 slika u pitanju

    
    cifar_batches = list(map(load_batch, range(1, 6)))
    # ovo je lista tuplova (images, label)
    all_images = np.concatenate(list(map(lambda b: b[0], cifar_batches)), axis=0)
    # sada pravimo listu od svih samo image, axis=0 da bi bilo po broju slika naci 50 000 za 5 batchova
    all_labels = reduce(lambda a, b: a + b, map(lambda b: b[1], cifar_batches))
    # map uzima sve label iz tuplova, reduce sluzi da sabira sve podnizove svakog batcha na prosli tako da bue 50 000 labela

    
    parovi = list(map(lambda i: (all_labels[i], all_images[i]), range(moj_len(all_labels))))
    # pravim parove sada za ovu od 50 000
    jedinstvene_klase = set(map(lambda par: par[0], parovi))
    # vadimo duplikate pomocu set
    average_histograms = list(map(lambda klasa: (klasa, prosecni_histogram(klasa, parovi)), jedinstvene_klase))
    # pravi listu tuplova jedinstvenih klasa i njihovig prosecnih histograma
    
    
    average_histograms_named = list(map(lambda klasa: (class_names[klasa], prosecni_histogram(klasa, parovi)), jedinstvene_klase))
    
    with open(output_file, "w") as f:
        
      f.writelines(
        map(
            lambda par: f"Class {par[0]}:\n{par[1].tolist()}\n",
            average_histograms_named
        )
      )
    print(f"Prosecni histogrami sacuvani kao {output_file}")
    
    return average_histograms

def cosine_similarity(hist1, hist2):
    
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()
   
    dot_product = reduce(lambda acc, pair: acc + pair[0] * pair[1], zip(flat_hist1, flat_hist2), 0)
    norm1 = np.sqrt(reduce(lambda acc, x: acc + x**2, flat_hist1, 0))
    norm2 = np.sqrt(reduce(lambda acc, x: acc + x**2, flat_hist2, 0))
    # samo ovaj deo izmenio od originala zbog specifikacije koja zahteva funkcionalni pristup
    # dot_product sada racunamo tako sto uzimamo parove pomocu zip
    # iteriramo po tim parovima i sabiramo ih jer dot_product = suma(pair[0] * pair[1])
    # norm = koren(suma(i^2)) tako da je jasna izmena
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

def klasifikator(image_path, average_histograms):
    
    image_histogram = racunaj_histogram_1(image_path)
    # uradim histogram za neku sliku
    
    similarities = map(
        lambda class_avg: (class_avg[0], cosine_similarity(image_histogram, class_avg[1])),
        average_histograms
    )
    # pravim iterable koji se sastoji od tuplova (klasa, slicnost) za slicnost prosecnog histograma svake klase i jedne slike 
    
    best_class = reduce(
        lambda max, x: max if max[1] > x[1] else x,
        similarities
    )
    # uzmimam najvecu slicnost

    #return (image_path, best_class[0], best_class[1])
    # ALTERNATIVNO ako treba kao broj
    # vracamo trazeno po specifikaciji kao tuple (slika, klasa, slicnost)
    
    best_class_name = class_names[best_class[0]]
    return (image_path, best_class_name, best_class[1])

'''
def save_image(image_data, save_path):
    
    image = Image.fromarray(image_data)
    image.save(save_path)

def extract_and_save_image(cifar_folder, batch_index, image_index, save_path):
    
    def load_batch(cifar_folder, batch_index):
        batch_file = os.path.join(cifar_folder, f"data_batch_{batch_index}")
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data']  
            labels = batch[b'labels']  
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
        return images, labels

    images, labels = load_batch(cifar_folder, batch_index)
    image_data = images[image_index]  
    
    save_image(image_data, save_path)
    print(f"Slika sacuvana {save_path}")
'''
# za cuvanje slika iz batcheva

def main():

    image_path = "test_image.jpg"  
    print("Histogram za jednu sliku:")
    histogram = racunaj_histogram_1(image_path)
    print(histogram)
    
    cifar_folder = "./cifar_batches/"
    load_class_names(cifar_folder)
    output_file = "prosecni_histogram.txt"  
    average_histograms = ucitaj_cifar(cifar_folder, output_file)
    
    hist1 = average_histograms[0][1]  
    hist2 = average_histograms[1][1]
    similarity = cosine_similarity(hist1, hist2)
    print(f"Kosinusna slicnost histograma prve i druge klase: {similarity}")
    
    zaKlasifikaciju = "airplane1.png"
    #zaKlasifikaciju = "deer6.png" 
    result = klasifikator(zaKlasifikaciju, average_histograms)
    print(f"Predvidjena klasa za sliku {result[0]}: {result[1]} a slicnost je {result[2]}")
    
    '''
    batch_index = 1  
    image_index = 0
    save_path = "saved_image.png"  
    extract_and_save_image(cifar_folder, batch_index, image_index, save_path)
    '''
    # poziv za cuvanje slika iz batcheva

if __name__ == "__main__":
    main()