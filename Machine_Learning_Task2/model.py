import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)
MODEL_SAVE_PATH = 'model/model-{epoch:03d}.h5'


def load_data(args):
    # dakle ovde se ucitavaju podaci
    # dele se u training, validation i testing
    data_df = pd.read_csv(
        os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'),
        names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    )
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.test_size, random_state=0
    )
    return X_train, X_valid, y_train, y_valid

# nvidijin fazon za kola
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def build_model(args):
    model = Sequential()
    # prvo cemo da normalizujemo input
    # vrednosti piksela idu u [-1, 1]
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))

    # dodajemo konvolucione slojeve
    # ovo sljaka dobro za analizu slika
    # model treba da "vidi" kako izgleda traka

    # strajdovi se koriste za downsampling slike na pocetku
    # nisu svi pikseli slike bitni u prvih par slojeva
    # exponential linear unit aktivacija
    # elu(x) = x,             x >= 0
    #          a * (e^x - 1), x < 0
    # elu se koristi da ne dovede do mrtvih neurona koje bi relu napravio
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))

    # nakon nekolko downsampled slojeva uzimamo celu sliku radi detalja
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # visedimenzionalni izlaz iz konvolucionih slojeva stavljamo u jedan niz
    # pre nego sto se salje u dense slojeve
    model.add(Flatten())

    # i onda trpamo dense slojeve
    model.add(Dense(100, activation='elu'))
    # dropout je za sprecavanje overfittinga
    # samo se gasi nasumicnih 50% neurona u svakoj iteraciji
    # valjalo bi probati sa drugacijim vrednostima ovoga
    # onda mozda nece da skace u jezero brat moj napaceni
    model.add(Dropout(0.24))
    # svaki naredni sloj ima manje neurona
    # jer se sprema za output sloj od jednog
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # i sad output sloj sa jednim neuronom - ugao volana
    model.add(Dense(1))

    # mean squared error + adam
    optimizer = Adam(lr=args.learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    # pravimo generatore za training i validation podatke
    train_gen = batch_generator(
        args.data_dir, X_train, y_train, args.batch_size, True
    )
    valid_gen = batch_generator(
        args.data_dir, X_valid, y_valid, args.batch_size, False
    )

    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))

    # model se cuva nakon svake epohe
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=True, # idemo bre
        mode='auto',
        period=1
    )

    # broj koraka u validaciji zavisi od velicine skupa
    val_steps = len(X_valid) // args.batch_size

    # i treniramo
    model.fit_generator(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.nb_epoch,
        validation_data=valid_gen,
        validation_steps=val_steps,
        callbacks=[checkpoint],
        verbose=1
    )


def main():
    # getopt_long je bolji
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',              dest='data_dir',          type=str,   default='data')
    parser.add_argument('-n', help='number of epochs',            dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='number of batches per epoch', dest='steps_per_epoch',   type=int,   default=100)
    parser.add_argument('-b', help='batch size',                  dest='batch_size',        type=int,   default=128)
    parser.add_argument('-l', help='learning rate',               dest='learning_rate',     type=float, default=1.0e-3)
    parser.add_argument('-t', help='validation split fraction',   dest='test_size',         type=float, default=0.2)
    args = parser.parse_args()
    
    X_train, X_valid, y_train, y_valid = load_data(args)
    model = build_model(args)

    train_model(model, args, X_train, X_valid, y_train, y_valid)

if __name__ == '__main__':
    main()
