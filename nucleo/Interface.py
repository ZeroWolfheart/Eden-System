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


class Analizador:
    
    def __init__(self):
        self._clases = []    # Clases que conoce el modelo. En orden.
        self._configuracion = None # Aqui debe estar el objeto de configuracion, iniciado con una ruta de archivo
        self._red = None # Aqui debe estar el objeto de red, iniciado con el valor "modelo" (archivo de modelo)
        self._imagen = None # objeto numpy con las matrices de la imagen.
    
    def cargar_Clases(self, lista=[]):
        ban = len(lista)>0
        if ban:
            for item in lista:
                self._clases.append(item)
        return ban
        
    def cargar_Configuracion(self, ruta=""):
        ban = ruta!=""
        if ban:
            self._configuracion = Configuracion(archivo=ruta)
        return ban
    
    def cargar_Red(self, ruta=""):
        ban = ruta!=""
        if ban:
            self._red = Red(configuracion=self._configuracion, modelo=ruta)
        return ban

    def cargar_Imagen(self, ruta=""):
        ban = ruta!=""
        if ban:
            self._imagen = skimage.io.imread(ruta)
            # Si esta en escala de grises, convertir en RGB para mantener consistencia
            if self._imagen.ndim != 3:
                self._imagen = skimage.color.gray2rgb(self._imagen)
            # Sí tiene un canal Alpha, remover para mantener consistencia
            if self._imagen.shape[-1] == 4:
                self._imagen = self._imagen[..., :3]
        return ban

    def analizar_Imagen(self):
        x = numpy.zeros((1,self._configuracion.FORMA_IMAGEN[0], self._configuracion.FORMA_IMAGEN[1], self._configuracion.FORMA_IMAGEN[2]))
        x[0], _ventana, _escala, _relleno, _aleatorio = Utiles.reescalar_Imagen(self._imagen, 
                                                                           minDim=self._configuracion.MIN_DIM, 
                                                                           maxDim=self._configuracion.MAX_DIM, 
                                                                           minEscala=self._configuracion.ESCALA_MINIMA, 
                                                                           modo=self._configuracion.MODO_REESCALADO)
        if self._configuracion.RED_TIPO_SALIDA in ['Y','L']:
            prediccion, deltas = self._red.red_neuronal.predict(x)
            anchors_Propuestos, deltas_Calculados, clases_Anchor = Modelo.decodificar_Tensores(
                t1=prediccion[0],
                t2=deltas[0],
                forma_Imagen=(self._configuracion.FORMA_IMAGEN[0], self._configuracion.FORMA_IMAGEN[1]),
                S=self._configuracion.S,
                B=self._configuracion.B,
                usar_Idf=self._configuracion.USAR_IDF
                )
        elif self._configuracion.RED_TIPO_SALIDA == 'I':
            prediccion = self._red.red_neuronal.predict(x)
            anchors_Propuestos, clases_Anchor = Modelo.decodificar_Unico_Tensor_Salida(
                t1=prediccion[0],
                S=self._configuracion.S,
                B=self._configuracion.B,
                forma_imagen=(self._configuracion.FORMA_IMAGEN[0], self._configuracion.FORMA_IMAGEN[1])
            )
        
        
        respuesta = self._imagen
        # Dibujar malla en imagen
        if self._configuracion.RESP_MALLA:
            mx,my = respuesta.shape[0]//self._configuracion.S, respuesta.shape[1]//self._configuracion.S
            color_malla = [255,255,255]
            respuesta[:,::my,:] = color_malla
            respuesta[::mx,:,:] = color_malla
        
        # Imprimir imagen (ventana contenedora)
        _, ax = pyplot.subplots(1, figsize=(16,16))
        altura, base = respuesta.shape[:2]
        ax.set_ylim(altura + 10, -10)
        ax.set_xlim(-10, base + 10)
        ax.axis('off')
        ax.set_title("Resultado")

        # Imprimir anchors_propuestos, deltas_calculados, clases_anchor
        # Para cada anchor
        for i in range(0,len(anchors_Propuestos)):
            # Valores de la caja (tensor 1)
            y1, x1, y2, x2, iou = anchors_Propuestos[i]
            # Si la red tiene dos tensores
            if self._configuracion.RED_TIPO_SALIDA in ['Y','L']:
                # Si esta activado el identificador de anchor positivo
                if self._configuracion.USAR_IDF:
                    dy,dx,logdh,logdw,idf = deltas_Calculados[i]
                    #TODO: Expiremental...
                    iou=idf
                else:
                    dy,dx,logdh,logdw = deltas_Calculados[i]
                # Funcion que unifica los valores devueltos por ambos
                # tensores en las coordenadas de la caja
                y1,x1,y2,x2=Utiles.aplicar_Delta_Caja(caja=[y1,x1,y2,x2],delta=[dy,dx,logdh,logdw])

            # Umbral de detección
            if iou > self._configuracion.DETECCION_CONFIDENCIA_MINIMA:
                # Generar color de la caja
                r,g,b = rn.random(), rn.random(), rn.random()
                # Selccionar clase a la que pertenece el elemento
                cc = numpy.argmax(clases_Anchor[i])
                
                #TODO: Comentar estas 3 lineas
                print([y1,x1,y2,x2])
                print(clases_Anchor[i])
                print(self._clases[cc])
                # Crear caja
                box = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
                ax.add_patch(box)
                # Formato de % de confidencia
                porc  = "{0:.2f}".format(iou*100)
                # Crear etiqueta de la caja
                ax.text(x1+2,y1 + 8, "{}%: {}".format(porc, self._clases[cc]), color=(r,g,b), size=11, backgroundcolor="none")
        
        # Crear e imprimir imagen en el contenedor
        pyplot.imshow(respuesta)
        pyplot.show()