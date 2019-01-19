import keras as k
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.callbacks import History
from keras.layers import Activation
from keras.models import model_from_json
from keras.optimizers import Adam


def my_autoencode(img_shape, code_size=32):
    H,W,C = img_shape
    
    # encoder
    encoder = k.models.Sequential()
    encoder.add(k.layers.InputLayer(img_shape))
    encoder.add(k.layers.Conv2D(32, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    encoder.add(k.layers.AveragePooling2D((2, 2), padding='same'))
    encoder.add(k.layers.Flatten())
    encoder.add(k.layers.Dense(512, activation='elu'))
    encoder.add(k.layers.Dense(256, activation='elu'))
    encoder.add(k.layers.Dense(code_size, activation='elu'))
    encoder.summary()

    # decoder
    decoder = k.models.Sequential()
    decoder.add(k.layers.InputLayer((code_size,)))
    decoder.add(k.layers.Dense(256, activation='elu'))
    decoder.add(k.layers.Dense(512, activation='elu'))
    decoder.add(k.layers.Dense(8192, activation='elu'))
    decoder.add(k.layers.Reshape((8, 8, 128)))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.UpSampling2D((2, 2)))
    decoder.add(k.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation='elu', padding='same'))
    decoder.add(k.layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='elu', padding='same')) # Unsure about this
    decoder.summary()
    
    return encoder, decoder