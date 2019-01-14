#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Obtencion del Path del conjunto de entrenamiento
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Obtencion de los nombres de las clases de entrenamiento y almacenarlos en una lista
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Obtencion de Path a las imagenes y guardado en una lista image_paths y la etiqueta correspondiente en image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Crear extraccion de caracteristicas y objetos detectores de puntos clave fea_det = cv2.FeatureDetector_create ("SIFT")
sift = cv2.xfeatures2d.SIFT_create()
#des_ext = cv2.DescriptorExtractor_create("SIFT")
#(kps, descs) = sift.detectAndCompute(gray, None)
# Lista donde todos los descriptores son almacenados.
des_list = []

cont_1 = 0
for image_path in image_paths:
    print ("cont_1: " + str(cont_1))
    im = cv2.imread(image_path)
    #kpts = fea_det.detect(im)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))
    cont_1 += 1   
    
# Apila todas los descriptores verticalmente en una matriz numpy
descriptors = des_list[0][1]
cont_2 = 0
for image_path, descriptor in des_list[1:]:
    print ("cont_2: " + str(cont_2))
    descriptors = np.vstack((descriptors, descriptor))
    cont_2 += 1  

k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculo de las caracteristicas del histograma
im_features = np.zeros((len(image_paths), k), "float32")
cont_3 = 0
for i in range(len(image_paths)):
    print ("cont_3: " + str(cont_3))
    cont_3 += 1	
    words, distance = vq(des_list[i][1],voc)
    cont_4 = 0
    for w in words:
        print ("cont_4: " + str(cont_4))
        im_features[i][w] += 1
        cont_4 += 1

nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')


stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Entrenamiento del SVM lineal
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Guardado de SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)    

# Abrir consola en el directorio principal del programa y tipear:  python findFeatures.py -t dataset/train/
