
"""
Script utilizado para entrenar un modelo 
convolucional con dataset de train/validation/test
que se encuentran en la carpeta data:


data/
    train/
        fake/
        real/
    test/
        fake/
        real/
    validation/
        fake/
        real/
"""
import os
import numpy as np
import tensorflow as tf
from dvclive import Live
from dvclive.keras import DVCLiveCallback

# Seleccionamos una semilla para los RNG
tf.random.set_seed(123)
np.random.seed(123)
main_dir = 'data/external/'
processed_dir = 'data/processed/'


# Directorio con las imágenes de fake 
fake_dir = os.path.join(main_dir, 'fake')

# Directorio con las imágenes de real 
real_dir = os.path.join(main_dir, 'real')
# The directory to save training images
train_dir = os.path.join(processed_dir, 'train')

# The directory to save validation images
val_dir = os.path.join(processed_dir, 'validation')


# The directory to save validation images
test_dir = os.path.join(processed_dir, 'test')

# Definimos un modelo en keras
conv_net = tf.keras.models.Sequential()

# Definimos una capa de entrada, especificamos que el tamaño de nuestras
# imágenes será de 150x150 y que tendrá tres canales.
conv_net.add(tf.keras.layers.Input(shape=(150, 150, 3)))

# Agregamos bloques de convolución seguidos de max pooling
# primer bloque
conv_net.add(tf.keras.layers.Conv2D(filters=36, kernel_size=3,
                                    activation='relu'))
conv_net.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# segundo bloque
conv_net.add(tf.keras.layers.Conv2D(filters=36, kernel_size=3,
                                    activation='relu'))
conv_net.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# tercer bloque
conv_net.add(tf.keras.layers.Conv2D(filters=36, kernel_size=3,
                                    activation='relu'))
conv_net.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# Agregamos una capa de flatten (transforma cualquier arreglo multidimensional 
# en un vector unidimensional)
# Convertir en un tensor de 1-dim
conv_net.add(tf.keras.layers.Flatten())

# Agregamos un clasificador (red neuronal multicapa)

# capa densa intermedia
conv_net.add(tf.keras.layers.Dense(512, activation='relu'))
# capa de salida
conv_net.add(tf.keras.layers.Dense(1, activation='sigmoid'))

conv_net.summary()
tf.keras.utils.plot_model(conv_net,show_shapes=True)
# Definimos las transformaciones para el conjunto de train
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Definimos las transformaciones para el conjunto de test
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# Especificamos el tamaño del batch, número de imagenes que genera en cada iteración 
batch_size = 128

# Definimos las transformaciones para el conjunto de test
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# Especificamos el tamaño del batch, número de imagenes que genera en cada iteración 
batch_size = 32


# Obtenemos un generador que realiza las transformaciones y carga
# las imágenes de entrenamiento
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

# Obtenemos un generador que realiza las transformaciones y carga
# las imágenes de validación
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                       target_size=(150, 150),
                                                       batch_size=batch_size,
                                                       class_mode='binary')

# Obtenemos un generador que realiza las transformaciones y carga
# las imágenes de validación
test_generator = test_datagen.flow_from_directory(test_dir,
                                                       target_size=(150, 150),
                                                       batch_size=batch_size,
                                                       class_mode='binary')
with open("models/metrics.txt", "w") as f:
    f.write("Test loss: {}\n".format(0))
    f.write("Test accuracy: {}\n".format(0))

# compilamos el modelo
conv_net.compile(loss="binary_crossentropy", optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 metrics=["accuracy"])
# entrenamos el modelo
with Live(save_dvc_exp=True) as live:
  history = conv_net.fit_generator(train_generator,
                        steps_per_epoch=2000//batch_size,  
                        epochs=20,
                        validation_data=validation_generator,
                        validation_steps=1000//batch_size,  
                        verbose=1,
                        callbacks=[
            DVCLiveCallback(save_dvc_exp=True, live=live)
        ])
  conv_net.save('models/model.h5')
  live.log_artifact("modelconv_net", type="model")

scores = conv_net.evaluate(test_generator)

# Escribimos las métricas en un archivo de texto
with open("models/metrics.csv", "w") as f:
    f.write("Test loss: {}\n".format(scores[0]))
    f.write("Test accuracy: {}\n".format(scores[1]))
