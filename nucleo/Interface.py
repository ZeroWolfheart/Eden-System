import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras as K
import keras.utils as KU
import keras.models as KM
import keras.layers as KL
import keras.optimizers as KO
import keras.callbacks as KC

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
import Visual
from Configuracion import Configuracion
from Red import Red
from Dataset import Dataset


class Analizador:
    
    def __init__(self):
        self._clases = []           # Clases que conoce el modelo. En orden.
        self._configuracion = None  # Aqui debe estar el objeto de configuracion, iniciado con una ruta de archivo
        self._red = None            # Aqui debe estar el objeto de red, iniciado con el valor "modelo" (archivo de modelo)
        self._imagen = None         # objeto numpy con las matrices de la imagen.
    
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
            self._red = Red(configuracion=self._configuracion)
            self._red.red_neuronal.load_weights(ruta)
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
            deltas_Calculados = None
            prediccion = self._red.red_neuronal.predict(x)
            anchors_Propuestos, clases_Anchor = Modelo.decodificar_Unico_Tensor_Salida(
                t1=prediccion[0],
                S=self._configuracion.S,
                B=self._configuracion.B,
                forma_imagen=(self._configuracion.FORMA_IMAGEN[0], self._configuracion.FORMA_IMAGEN[1])
            )
        
        
        respuesta = self._imagen
        # Dibujar malla en imagen
        respuesta = Visual.dibujar_Malla(
            configuracion=self._configuracion, imagen=respuesta
        )
        # Imprimir imagen (ventana contenedora)
        Visual.imprimir_Predicciones(imagen = respuesta, configuracion = self._configuracion,
                          anchors_Propuestos = anchors_Propuestos, deltas_Calculados = deltas_Calculados,
                          clases_Anchor = clases_Anchor, clases = self._clases)
        

class Entrenador:

    def __init__(self):
        self._configuracion = None  # Objeto de configuración utilizado por la red
        self._red = None            # Objeto de red, que se entrenará
        self._dataSet = None        # Objeto de la clase DataSet, contiene la lista de clases
        self._epoca_Inicio = 0      # Epoca en la que se inicia el entrenamiento
        self._epocas_Entrenamiento = 0 # Cantidad de epocas a entrenar
        
    def cargar_Configuracion(self, ruta=""):
        ban = ruta!=""
        if ban:
            self._configuracion = Configuracion(archivo=ruta)
        return ban
    
    def cargar_Red(self, ruta=""):
        ban = ruta!=""
        if ban:
            self._red = Red(configuracion=self._configuracion)
            self._red.red_neuronal.load_weights(ruta)
            self._verificar_Epoca_Inicial(ruta=ruta)
        else:
            self._red = Red(configuracion=self._configuracion)
        return ban
    
    def cargar_Dataset(self, ruta="", nombre="", clases=[]):
        ban = ruta!="" and nombre!="" and len(clases)>0
        if ban:
            self._dataSet = Dataset(nombre,ruta)
            for elemento in clases:
                self._dataSet.agregar_Clase(elemento)
            self._dataSet.cargar_Dataset()
            self._dataSet.crear_SubSets()
        return ban
    
    def establecer_Epocas_Entrenamiento(self, cantidad=0):
        self._epocas_Entrenamiento = int(cantidad)
    
    def entrenar_Red(self):
        # ajustar parametros de decenso de gradiente (Stochastic Gradient Descent)
        # Decesnso de Gradiente Estocastico
        sgd = KO.SGD(lr=self._configuracion.APRENDIZAJE_TASA,
                     momentum=self._configuracion.APRENDIZAJE_MOMENTO,
                     decay=self._configuracion.PESO_PERDIDA)
        # Compilar la red con el optimizador anterior y error cuadratico medio
        self._red.red_neuronal.compile(optimizer=sgd, loss='mean_squared_error')
        # Crear generador de entrenamiento
        gen_Entrenamiento =  Modelo.generador_Datos(
            dataset=self._dataSet,
            modo="entrenamiento",
            configuracion=self._configuracion,
            revolver=True,
            tam_batch=self._configuracion.TAM_BATCH
        )
        # Crear generador de Válidacion
        gen_Validacion = Modelo.generador_Datos(
            dataset=self._dataSet,
            modo="validacion",
            configuracion=self._configuracion,
            revolver=True,
            tam_batch=self._configuracion.TAM_BATCH
        )
        # Calcular pasos para entrenamiento y  validacion
        paso_Entrenamiento = math.ceil(len(self._dataSet.entrenamiento)/self._configuracion.TAM_BATCH)
        paso_Validacion = math.ceil(len(self._dataSet.validacion)/self._configuracion.TAM_BATCH)
        # Crear los directorios que contienen los resultados de entrenamiento
        self._crear_Directorios()
        # Configurar entrenador para usar directorios
        self._configurar_Directorios()
        # Callbacks
        # Crear callback que elimina el exceso de  archivos de pesos
        auto_borrador = KC.LambdaCallback(
            on_epoch_end=lambda epoch, logs: self._borrador("pesos/{}/".format(self._dataSet.nombre_Dataset))
            )
        # Crear arreglo de callbacks (requerido por Keras)
        callbacks = []
        if self._configuracion.ENTRENAMIENTO_LOGS:
            callbacks.append(KC.CSVLogger(self._ruta_Logs, separator=',', append=True))
        callbacks.append(KC.ModelCheckpoint(self._ruta_Pesos, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False))
        callbacks.append(auto_borrador)
        # Entrenar (Uso del metodo de Keras para entrenar la red neuronal)
        self._red.red_neuronal.fit_generator(
            gen_Entrenamiento,                      # Generador de datos de entrenamiento
            initial_epoch=self._epoca_Inicio,      # Epoca en la que se inicia
            epochs=self._epocas_Entrenamiento+self._epoca_Inicio,     # Cantidad de epocas  que se entrenará
            steps_per_epoch=paso_Entrenamiento,    # Pasos que ejecutara en cada epoca de entrenamiento
            callbacks= callbacks,                   # Callbacks que ejecutara al final de cada epoca
            validation_data=gen_Validacion,         # Generador de  datos de Validación
            validation_steps=paso_Validacion,      # Pasos que ejecutará en cada epoca de validación
            max_queue_size=self._configuracion.COLA,# Tamaño de la fila del generador (keras lo establece en 10 por defecto)
            workers=self._configuracion.TRABAJADORES, # Hilos en los que se dividirá el proceso                               
            use_multiprocessing=self._configuracion.USAR_MULTI, # Activar si existe mas de 1 procesador grafico
            verbose=2                               # Modo de imprimir progreso
        )

    def _crear_Directorios(self):
        # Crear directorio de pesos si no existe
        if not os.path.exists("pesos/{}".format(self._dataSet.nombre_Dataset)):
            os.makedirs("pesos/{}".format(self._dataSet.nombre_Dataset))
        # Crear directorio de logs si no existe
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def _configurar_Directorios(self):
        # Configurar ruta de guardado de pesos
        self._ruta_Pesos = os.path.join("pesos/{}".format(self._dataSet.nombre_Dataset), "epoca_*epoch*.h5")
        self._ruta_Pesos = self._ruta_Pesos.replace("*epoch*", "{epoch:04d}")
        if self._configuracion.ENTRENAMIENTO_LOGS:
            self._ruta_Logs = os.path.join("logs", "entrenamiento_{}.log".format(self._dataSet.nombre_Dataset))

    def _borrador (self, direcccion):
        # listar elementos en la ruta
        saves = os.listdir(direcccion)
        # borrar si la cantidad es mayor a la establecida en configuracion
        if len(saves) > self._configuracion.GUARDADOS_MAX:
            saves = sorted(saves, reverse=True)
            borrables = saves[self._configuracion.GUARDADOS_MAX:]
            for borrable in borrables:
                os.remove(direcccion+borrable)
                
    def _verificar_Epoca_Inicial(self,ruta=""):
        ruta = ruta.split("_")
        epoca = ruta[-1].split(".")
        self._epoca_Inicio = int(epoca[0])
    
    def info_Red(self):
        self._red.sumarizar_Red()
        self._red.red_a_IMG()