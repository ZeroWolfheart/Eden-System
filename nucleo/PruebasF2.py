import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras as K
import keras.utils as KU
import keras.models as KM
import keras.layers as KL
import numpy
import skimage.color
import skimage.io
import skimage.transform

from matplotlib import pyplot
from matplotlib.patches import Rectangle
import random as rn
import math

import Utiles
import Modelo
from Configuracion import Configuracion
from Red import Red

calses = []
# calses.append("kangaroo")

calses.append("apple")
calses.append("banana")
calses.append("orange")

# calses.append("aeroplane")
# calses.append("bicycle")
# calses.append("bird")
# calses.append("boat")
# calses.append("bottle")
# calses.append("bus")
# calses.append("car")
# calses.append("cat")
# calses.append("chair")
# calses.append("cow")
# calses.append("diningtable")
# calses.append("dog")
# calses.append("horse")
# calses.append("motorbike")
# calses.append("person")
# calses.append("pottedplant")
# calses.append("sheep")
# calses.append("sofa")
# calses.append("train")
# calses.append("tvmonitor")



miConfig = Configuracion()
#miRed =  Red(configuracion=miConfig)
dirr = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'pesos'))
miRed = Red(modelo=dirr+"/Eden_SystemV4_Frutero_13751.h5")
#miRed.red_neuronal.load_weights("pesos/Eden_SystemV4_Frutero_0276.h5")
#modelo = KM.load_model("pesos/Eden_SystemV4_Frutero_0276.h5")
dirr = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'test'))
imagen = skimage.io.imread(dirr+"/manzana-macintosh.jpg")
# Si esta en escala de grises, convertir en RGB para mantener consistencia
if imagen.ndim != 3:
    imagen = skimage.color.gray2rgb(imagen)
# Sí tiene un canal Alpha, remover para mantener consistencia
if imagen.shape[-1] == 4:
    imagen = imagen[..., :3]

imagen, _ventana, _escala, _relleno, _aleatorio = Utiles.reescalar_Imagen(imagen, 
                                                                           minDim=miConfig.MIN_DIM, 
                                                                           maxDim=miConfig.MAX_DIM, 
                                                                           minEscala=miConfig.ESCALA_MINIMA, 
                                                                           modo = "cuadrado")

input = numpy.zeros((1,miConfig.FORMA_IMAGEN[0],miConfig.FORMA_IMAGEN[1],miConfig.FORMA_IMAGEN[2]))
input[0] = imagen
print(imagen.shape)

if  miConfig.RED_TIPO_SALIDA in ['Y','L']:
    prediction, deltas = miRed.red_neuronal.predict(input)
    #modelo.predict(input)
elif miConfig.RED_TIPO_SALIDA == "I":
    prediction = miRed.red_neuronal.predict(input)
    # modelo.predict(input)
print(prediction[0])

if  miConfig.RED_TIPO_SALIDA in ['Y','L']:
    anchors_propuestos, deltas_calculados, clases_anchor =  Modelo.decodificar_Tensores(t1=prediction[0],
                                                                                    t2=deltas[0],
                                                                                    forma_Imagen=(miConfig.FORMA_IMAGEN[0], miConfig.FORMA_IMAGEN[1]),
                                                                                    S=miConfig.S,
                                                                                    B=miConfig.B,
                                                                                    usar_Idf=miConfig.USAR_IDF)
elif miConfig.RED_TIPO_SALIDA == "I":
    anchors_propuestos, clases_anchor = Modelo.decodificar_Unico_Tensor_Salida(t1=prediction[0],
                                                                               S=miConfig.S,
                                                                               B=miConfig.B,
                                                                               forma_imagen=(miConfig.FORMA_IMAGEN[0], miConfig.FORMA_IMAGEN[1]))
# Forma de imprimir
mx,my = imagen.shape[0]//miConfig.S, imagen.shape[1]//miConfig.S
color_malla=[255,255,255]
imagen[:,::my,:] = color_malla
imagen[::mx,:,:] = color_malla

_, ax = pyplot.subplots(1, figsize=(16,16))
height, width = imagen.shape[:2]
ax.set_ylim(height + 10, -10)
ax.set_xlim(-10, width + 10)
ax.axis('off')
ax.set_title("el titutlo")

#imprimir anchors_propuestos, deltas_calculados, clases_anchor
for i in range(0,len(anchors_propuestos)):
    y1, x1, y2, x2, iou = anchors_propuestos[i]
    # if miConfig.USAR_IDF:
    #     dy,dx,logdh,logdw,idf = deltas_calculados[i]
    # else:
    #     dy,dx,logdh,logdw = deltas_calculados[i]
    
    r,g,b = rn.random(), rn.random(), rn.random()
    cc = numpy.argmax(clases_anchor[i])
    # indicador
    if iou > 0.00:
        zz=3
        # print(dy,dx,logdh,logdw)
        # deltx =  Utiles.aplicar_Delta_Caja(caja=[y1,x1,y2,x2],delta=[dy,dx,logdh,logdw])
        # print(deltx)
        print([y1,x1,y2,x2])
        print(clases_anchor[i])
        print(calses[cc])
        # amr2 = Rectangle((deltx[1],deltx[0]), deltx[3]-deltx[1], deltx[2]-deltx[0], linewidth=zz, alpha=0.7, linestyle='dashed', edgecolor=(r,g,b), facecolor='none')
        # ax.add_patch(amr2)
        
        amr = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr)
        
        perc  = "{0:.2f}".format(iou*100)
        
        ax.text(x1+2,y1 + 8, "{}%: {}".format(perc, calses[cc]), color=(r,g,b), size=11, backgroundcolor="none")
        
        
pyplot.imshow(imagen)
pyplot.show()


