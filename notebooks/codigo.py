#################################################################################
#################################################################################
#############             WILCOXON                               ################
#############                                                    ################
#################################################################################
#################################################################################
resultadosROCmodelo1 = []
resultadosROCmodelo2 = []

def RecogerDatosFichero(nombreModelo,lista):
    with open(f'datosROC{nombreModelo}.txt', 'r') as file:
        # Leer el contenido del archivo y dividirlo en elementos
        contenido = file.read().split()

        # Convertir los elementos a números (si es necesario)
        lista.extend(float(elemento) for elemento in contenido)

def wilcoxonTest(modelo1, modelo2):

    #Es el modelo 1 mejor que le mdelo 2
    stat, p_value = wilcoxon(modelo1, modelo2, alternative ='greater')

    # Imprime los resultados
    print(f'Estadístico de prueba: {stat}')
    print(f'Valor p: {p_value}')

    # Comprueba si la diferencia es estadísticamente significativa
    if p_value < 0.05:
        print('La diferencia es estadísticamente significativa.')
    else:
        print('No hay evidencia suficiente para afirmar que hay una diferencia significativa.')

def mainCompararValoresRoc():
    nombreModelo1 = "modelo0"
    nombreModelo2 = "modelo1"

    RecogerDatosFichero(nombreModelo1, resultadosROCmodelo1)
    RecogerDatosFichero(nombreModelo2, resultadosROCmodelo2)

    print(resultadosROCmodelo1)
    print(resultadosROCmodelo2)

    wilcoxonTest(resultadosROCmodelo1,resultadosROCmodelo2)

if __name__ == '__main__':
    mainCompararValoresRoc()

#################################################################################
#################################################################################
#############             DESAFIO 1                              ################
#############                                                    ################
#################################################################################
#################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers

from keras.layers import Dropout


#Wilcoxon Test
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import wilcoxon

#Utilizado para guardar modelos y cargarlos
import joblib

import shap

import cv2

from PIL import Image

#Entrenamiento
batch_size = 64
nb_classes = 3
epochs = 50
crossValidationSplit = 10
# Scaling input image to theses dimensions
img_rows, img_cols = 16, 16

nombreModelo = "modelo2GeneracionDeDatos3Clases16x16"

resultadosROC = []

###########################################################
###########################################################
#############             3Clases          ################
#############                              ################
###########################################################
###########################################################
def load_data():
  name_classes = ['NORMAL','PNEUMONIA_BACTERIA','PNEUMONIA_VIRUS']
  X,y  = [], []
  for class_number, class_name in enumerate(name_classes):    # Number of directories (0, 'NORMAL')
    for filename in glob.glob(f'./chest_xray_512/{class_name}/*.jpg'):
        im = preprocesar_imagen(filename)
        #im = image.load_img(filename, target_size=[img_rows, img_cols], color_mode = 'grayscale')
        X.append(image.img_to_array(im))
        y.append(class_number)

  input_shape = (img_rows, img_cols, 1)

  return np.array(X), np.array(y), input_shape

###########################################################
###########################################################
#############             2Clases          ################
#############                              ################
###########################################################
###########################################################
def load_data():
  name_classes = ['NORMAL','PNEUMONIA']
  X,y  = [], []
  for class_number, class_name in enumerate(name_classes):    # Number of directories
    for filename in glob.glob(f'./chest_xray_512/{class_name}/*.jpg'):
        im = preprocesar_imagen(filename)
        #im = image.load_img(filename, target_size=[img_rows, img_cols], color_mode = 'grayscale')
        X.append(image.img_to_array(im))
        y.append(class_number)

  input_shape = (img_rows, img_cols, 1)

  return np.array(X), np.array(y), input_shape

def preprocesar_imagen(imagen_path):
    # Cargar la imagen utilizando OpenCV
    imagen = cv2.imread(imagen_path)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar ajuste de brillo y contraste
    alpha = 1.3  # factor de contraste
    beta = -80    # factor de brillo
    imagen_ajustada = cv2.convertScaleAbs(imagen_gris, alpha=alpha, beta=beta)

    # Aplicar umbralización
    #_, imagen_umbralizada = cv2.threshold(imagen_ajustada, 127, 255, cv2.THRESH_BINARY)

    # Aplicar desenfoque
    #imagen_desenfocada = cv2.GaussianBlur(imagen_umbralizada, (5, 5), 0)

    # Aplicar ecualización del histograma
    #imagen_ecualizada = cv2.equalizeHist(imagen_desenfocada)

    # Realizar otras transformaciones si es necesario, como redimensionar, normalizar, etc.
    imagen_final = cv2.resize(imagen_ajustada, (img_rows, img_cols), interpolation=cv2.INTER_AREA)

    return imagen_final

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(n, 3))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i],:,:,0])
        plt.gray()
        ax.set_title('{}-{}'.format(y[index[i]],index[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

###########################################################
###########################################################
#############     3Clases/2Clases 32x32    ################
#############                              ################
###########################################################
###########################################################
def cnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(32, (2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(nb_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

###########################################################
###########################################################
#############             3Clases 16x16    ################
#############                              ################
###########################################################
###########################################################
def cnn_model(input_shape, nb_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(nb_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def GuardarValoresROCfichero():
    # Escribimos los resultado en un fichero de texto
    with open(f"datosROC{nombreModelo}.txt", 'w') as file:
        # Escribir los elementos en el archivo, separados por espacios
        file.write(' '.join(map(str, resultadosROC)))

##################################################################################
# Main program
def main():
    X, y, input_shape = load_data()

    print(X.shape, 'train samples')
    print(img_rows,'x', img_cols, 'image size')
    print(input_shape,'input_shape')
    print(epochs,'epochs')

    plot_symbols(X, y)
    collections.Counter(y)

    #Generar imagenes
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #CV-10
    kf = StratifiedKFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

        print("Número de imágenes generadas:", len(train_datagen) * batch_size)

        print(f'x_train {X_train.shape} x_test {X_test.shape}')
        print(f'y_train {y_train.shape} y_test {y_test.shape}')

        model = cnn_model(input_shape, nb_classes)
        print(model.summary())

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

        #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
        history = model.fit(train_datagen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

        # Obtener las métricas del historial
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Graficar la precisión
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # Graficar la pérdida
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

        #Visualizar datos del split
        loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        y_pred = model.predict(X_test)

        #Guardar datos ROC
        #resultadosROC.append(roc_auc_score(y_test, y_pred[:, 1]))
        resultadosROC.append(roc_auc_score(y_test, y_pred, multi_class='ovr'))

        #Graficas sobre Test
        print(f"Split numero {splitEntrenamiento}:")
        print(f'loss: {loss:.2f} acc: {acc:.2f}')
        #print(f'AUC {roc_auc_score(y_test, y_pred[:, 1], ):.4f}')
        print(f'AUC {resultadosROC[splitEntrenamiento-1]:.4f}')

        print('Predictions')
        y_pred_int = y_pred.argmax(axis=1)
        print(collections.Counter(y_pred_int), '\n')

        print('Metrics')
        print(metrics.classification_report(y_test, y_pred_int, target_names=['Normal', 'Pneumonia_bacteriana', 'Pneumonia_viral']))

        print('Confusion matrix')
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred_int),
                                       display_labels=['NORMAL', 'PNEUMONIA_BACTERIANA','PNEUMONIA_VIRAL']).plot()
        plt.show()


        #SHAP ver que partes de la imagen se fija el modelo 
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

        # Selecciona 10 imágenes al azar de X_test
        indices = np.random.choice(np.arange(len(X_test)), size=10, replace=False)
        X_explain = X_test[indices]

        # Usa solo un subset de X_train
        X_train_subset = X_train[:1000]
        explainer = shap.DeepExplainer(model, X_train_subset)

        # Calcula los valores SHAP
        shap_values = explainer.shap_values(X_explain)

        # Visualiza los valores SHAP
        shap.image_plot(shap_values, -X_explain)

        splitEntrenamiento += 1

    GuardarValoresROCfichero()

    print("Fin de entrenamiento")

if __name__ == '__main__':
    main()