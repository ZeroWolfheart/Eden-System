import numpy
from configparser import ConfigParser

class Configuracion:

    def __init__(self, archivo=None):
        config = ConfigParser()
        if archivo==None:
            config.read("config/default.conf")
        else:
            config.read(archivo)

    ##GPU##
        # Numero de GPU a utilizar.
        self.GPUs = config.getint("GPU","GPUs")
        # Numero de imagenes a entrenar en por GPU.
        # Una GPU de 12 GB puede manejar 2 imagenes de 1024*1024px
        # Ajustar con base a la memoria de la GPU a utilizar, siempre buscando el valor más alto
        self.IMG_GPU = config.getint("GPU","GPUs")
        # Tamaño de batch efectivo
        self.TAM_BATCH = self.IMG_GPU * self.GPUs

    ##IMAGEN##
        # Reescalado de imagen de entrada
        # Generalmente utilizar "cuadrado" tanto para entrenamiento como predicción
        # Modos de reescalado disponiles:
        #       none: regresa la imagen sin cambios
        #       cuadrado: Reescala y rellena con ceros para tener una imagen de maxDim * maxDim
        #       relleno64: rellena base y altura con zeros, para hacerlos multiplos de 64. Sí minDim o minEscala son provistos, escala la imagen antes de rellenar. maxDim es ignorado en este modo.
        #       aleatorio: toma fragmentos aleatorios de la imagen. Primero escala la imagen basado en minDim y minEscala, despues, toma un fragmento aleatorio de tamaño minDim * minDim
        self.MODO_REESCALADO = config.get("IMAGEN","MODO_REESCALADO")
        self.MIN_DIM = config.getint("IMAGEN","MIN_DIM")
        self.MAX_DIM = config.getint("IMAGEN","MAX_DIM")
        # Factor minimo de reescalado. Se verifica despues de MMIN_DIM y puede forzar
        # una escala mayor. Por ejemplo, si esta establecido en 2 las imagenes se escalarán al doble
        # de ancho y alto, aun si MIN_DIM no lo requiere.
        # Sin embargo, en modo 'cuadrado', puede ser sobre pasado por MAX_DIM.
        self.ESCALA_MINIMA = config.getint("IMAGEN","ESCALA_MINIMA")
        # Numero de canales por color. RGB = 3, escala de grises = 1, RGB-D = 4
        self.CANALES = config.getint("IMAGEN","CANALES")
        # Tamaño de imagen de entrada
        if self.MODO_REESCALADO == "aleatorio":
            self.FORMA_IMAGEN = numpy.array([self.MIN_DIM, self.MIN_DIM, self.CANALES])
        else:
            self.FORMA_IMAGEN = numpy.array([self.MAX_DIM, self.MAX_DIM, self.CANALES])

    ##MODELO##
        # Numero de clases a identificar (incluyendo background)
        self.NUM_CLASES = config.getint("MODELO","NUM_CLASES")
        # Celdas por lado de regilla
        # La imagen se divirá en una regilla de S x S celdas
        # Es importante para el calculo de las capas del perceptrón y el tensor de salida
        # Debe poder dividir los lados de la imagen de manera entera
        self.S = config.getint("MODELO","S")
        # Cantidad de Anchors (cajas predictivas), generas por cada celda de la regilla
        self.B = config.getint("MODELO","B")
        # Tipo de Red
        # Tipo de red que es construida.
        #TODO: Describir tipos de red
        #"Conv24"
        #"Conv19"
        #"Res55"
        self.RED_TIPO = config.get("MODELO","RED_TIPO")
        # Tipo de salida:
        # "Y": Construye una salida con forma de Y, que tiene su disyunción al finalizar la red
        #   convolucional y clacula por separado, los Anchors y deltas
        # "L": Contruye una salida con forma de L, donde la salida que calcula las deltas, se
        #   se encuentra como una derivación de la salida que calcula los Anchors
        # "I": Construye una unica salida, que solo contempla los Anchor y la desviación adecuada
        #   para aquel más apto
        self.RED_TIPO_SALIDA = config.get("MODELO","RED_TIPO_SALIDA")

    ##ANCLAS##
        # Longitud de un lado del Anchor cuadrado en pixeles
        self.ANCHOR_SCALAS = self._convertidor_Scalas(config.get("ANCLAS","ANCHOR_SCALAS"))
        # Factores de Anchors en cada celda (base/ altura)
        # Un valor 1 representa un Anchor cuadrado, y 0.5 es un Anchor más ancho (con mayor base)
        self.ANCHOR_FACTORES = self._convertidor_Factores(config.get("ANCLAS","ANCHOR_FACTORES"))
        # Si esta activado, reduce la instancia de la mascara para
        # reducir la carga de memoria. Recomendado cuando se usan imagenes de alta resulucion.
        self.USAR_MINIMASCARA = config.getboolean("ANCLAS","USAR_MINIMASCARA")
        self.MINIMASCARA_SHAPE = self._convertidor_Minimascara(config.get("ANCLAS","MINIMASCARA_SHAPE"))
        # Cantos Anchors se utilizarán para el entrenamiento en 1 imagen
        self.ANCHORS_ENTRENAMIENTO_IMAGEN = self.S*self.S*self.B

    ##APRENDIZAJE##
        # IoU Minimo para marcar como positivo un Anchor
        self.DELTA_IOU_MIN_POSITIVO = config.getfloat("APRENDIZAJE","DELTA_IOU_MIN_POSITIVO")
        # Umbral maximo para mmarcar como negativo un Anchor
        self.DELTA_IOU_MAX_NEGATIVO = config.getfloat("APRENDIZAJE","DELTA_IOU_MAX_NEGATIVO")
        # Utilizar o no el idf del algoritmo de generación de deltas
        self.USAR_IDF = config.getboolean("APRENDIZAJE","USAR_IDF")
        # Maximo numero de intancias (rerales) en una imagen
        self.MAX_R_INSTANCIAS = config.getint("APRENDIZAJE", "MAX_R_INSTANCIAS")
        # Max number of final detections
        self.DETECTION_MAX_INSTANCES = config.getint("APRENDIZAJE", "DETECTION_MAX_INSTANCES")
        # Non-maximum suppression threshold for detection
        self.DETECTION_NMS_THRESHOLD = config.getfloat("APRENDIZAJE", "DETECTION_NMS_THRESHOLD")
        # Learning rate and momentum
        # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        # weights to explode. Likely due to differences in optimizer
        # implementation.
        self.LEARNING_RATE = config.getfloat("APRENDIZAJE", "LEARNING_RATE")
        self.LEARNING_MOMENTUM = config.getfloat("APRENDIZAJE", "LEARNING_MOMENTUM")
        # Weight decay regularization
        self.WEIGHT_DECAY = config.getfloat("APRENDIZAJE", "WEIGHT_DECAY")
        # Desviacion estandar para el refoinamiento de la caja contenedora para entrenamiento y deteccion
        self.ENT_CC_STD_DEV =self._convertidor_Numpy1linea(config.get("APRENDIZAJE", "ENT_CC_STD_DEV")) #numpy
        self.BBOX_STD_DEV = self._convertidor_Numpy1linea(config.get("APRENDIZAJE", "BBOX_STD_DEV"))#numpy

    ##RESPUESTA##
        # Dibujar o no la malla en la imagen respuesta
        self.RESP_MALLA = config.getboolean("RESPUESTA","RESP_MALLA")
        # Probabilidad minima para aceptar una instancia detectada
        # instancia un valor menor a esta probabilidad seran omitidas
        self.DETECCION_CONFIDENCIA_MINIMA=config.getfloat("RESPUESTA","DETECCION_CONFIDENCIA_MINIMA")

## Metodos de clase (para leer valores de archivo)

    def _convertidor_Scalas(self, valores=""):
        valores = valores.splitlines()
        for i in range(len(valores)):
            valores[i]=valores[i].split(",")
            for j in range(len(valores[i])):
                valores[i][j]=float(valores[i][j])
        return valores

    def _convertidor_Factores(self, valores=""):
        valores = valores.split(",")
        for i in range(len(valores)):
            valores[i]=float(valores[i])
        return valores

    def _convertidor_Minimascara(self, valores=""):
        valores=valores.split(",")
        valores[0]=int(valores[0])
        valores[1]=int(valores[1])
        tupla=(valores[0],valores[1])
        return tupla

    def _convertidor_Numpy1linea(self, valores=""):
        lista=self._convertidor_Factores(valores)
        lista = numpy.array(lista)
        return lista