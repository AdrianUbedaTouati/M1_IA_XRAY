import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

def resize_image(imagen_path, modificacion, pixels):
    # Cargar la imagen utilizando OpenCV
    image = cv2.imread(imagen_path)

    # Convertir la imagen a escala de grises
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if modificacion:
        alpha = 1.3  # factor de contraste
        beta = -80    # factor de brillo
        image_gris = cv2.convertScaleAbs(image_gris, alpha=alpha, beta=beta)

    imagen_final = cv2.resize(image_gris, (pixels, pixels), interpolation=cv2.INTER_AREA)

    return imagen_final

def load_data(path ,modificacion, pixels):
    name_classes = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']
    X, y = [], []

    # Listar las carpetas principales
    base_dirs = [path + r'\val']

    for base_dir in base_dirs:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            path = rf'{base_dir}\{class_name}\*.jpeg'
            for filename in glob.glob(path):
                im = resize_image(filename, modificacion, pixels)
                X.append(image.img_to_array(im))
                
                # Determinar la clase en función del nombre del archivo
                if class_name == 'NORMAL':
                    y.append(0)  # Clase 0: NORMAL
                elif 'bacteria' in filename.lower():
                    y.append(1)  # Clase 1: PNEUMONIA_BACTERIA
                elif 'virus' in filename.lower():
                    y.append(2)  # Clase 2: PNEUMONIA_VIRUS

    input_shape = (pixels, pixels, 1)
    return np.array(X), np.array(y), input_shape

def main(path_model,path_images):
    # Cargar el modelo
    model = load_model(path_model)

    # Resumen del modelo
    model.summary()

    modificacion_32x32 = False
    pixels_32x32 = 32

    images_32x32 ,etiquetes_32x32 ,_ = load_data(path_images, modificacion_32x32, pixels_32x32)

    # Faire les prédictions
    predictions = model.predict(images_32x32)

    # Convertir les probabilités en étiquettes (pour un problème de classification binaire ou multi-classes)
    etiquettes_predites = np.argmax(predictions, axis=1)

    # Assurez-vous que les étiquettes sont dans le bon format
    etiquettes_reelles = np.array(etiquetes_32x32)

    # Calculer la précision
    precision = accuracy_score(etiquettes_reelles, etiquettes_predites)
    print(f"Précision du modèle : {precision:.2f}")

    # Rapport détaillé de classification
    print("Rapport de Classification :")
    print(classification_report(etiquettes_reelles, etiquettes_predites))

    # Sélection de quelques images à visualiser
    nombre_exemples = 6
    indices = np.random.choice(len(images_32x32), nombre_exemples, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(1, nombre_exemples, i + 1)
        plt.imshow(images_32x32[idx].squeeze(), cmap='gray')
        plt.title(f"Réel : {etiquettes_reelles[idx]}\nPrédit : {etiquettes_predites[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    path_model = r"models\meilleur_modele_roc_2_2.h5"
    path_images = r"data\external\chest_xray"
    main(path_model, path_images)
