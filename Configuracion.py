import numpy

class Configuracion:

    # Numero de GPU a utilizar.
    # GPUs = 1
    GPUs = 1

    # Numero de imagenes a entrenar en por GPU.
    # Una GPU de 12 GB puede manejar 2 imagenes de 1024*1024px
    # Ajustar con base a la memoria de la GPU a utilizar, siempre buscando el valor más alto
    # IMG_GPU = 2
    IMG_GPU = 1
    
    # Numero de clases a identificar (incluyendo background)
    NUM_CLASES = 1  # Sobre escribir en subclase

    # Longitud de un lado del Anchor cuadrado en pixeles
    ANCHOR_SCALAS =    [[ 21.68877 ,  27.029394],
                        [ 65.429688, 111.17647 ],
                        [ 44.92469 ,  41.75885 ],
                        [168.      , 119.549236],
                        [ 73.      , 169.269522],
                        [ 34.980114,  98.681446],
                        [136.      , 170.8     ],
                        [103.917694, 152.079476],
                        [ 89.09091 , 127.835052],
                        [ 69.775176,  69.309662],
                        [ 50.485602, 152.764692],
                        [105.777778,  91.2     ]]

    # Factores de Anchors en cada celda (base/ altura)
    # Un valor 1 representa un Anchor cuadrado, y 0.5 es un Anchor más ancho (con mayor base)
    ANCHOR_FACTORES =  [0.33, 0.35, 0.43, 0.59, 0.68, 0.7, 0.8, 0.8, 1.01, 1.08, 1.16, 1.41]

    # IoU Minimo para marcar como positivo un Anchor
    DELTA_IOU_MIN_POSITIVO = 0.4
    
    # Umbral maximo para mmarcar como negativo un Anchor
    DELTA_IOU_MAX_NEGATIVO = 0.3
    
    # Utilizar o no el idf del algoritmo de generación de deltas
    USAR_IDF = True

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Si esta activado, reduce la instancia de la mascara para
    # reducir la carga de memoria. Recomendado cuando se usan imagenes de alta resulucion.
    USAR_MINIMASCARA = True
    MINIMASCARA_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Reescalado de imagen de entrada
    # Generalmente utilizar "cuadrado" tanto para entrenamiento como predicción
    # Modos de reescalado disponiles:
    #       none: regresa la imagen sin cambios
    #       cuadrado: Reescala y rellena con ceros para tener una imagen de maxDim * maxDim
    #       relleno64: rellena base y altura con zeros, para hacerlos multiplos de 64. Sí minDim o minEscala son provistos, escala la imagen antes de rellenar. maxDim es ignorado en este modo.
    #       aleatorio: toma fragmentos aleatorios de la imagen. Primero escala la imagen basado en minDim y minEscala, despues, toma un fragmento aleatorio de tamaño minDim * minDim
    
    MODO_REESCALADO = "cuadrado"
    MIN_DIM = 128   #800
    MAX_DIM = 200  #1024
    
    # Factor minimo de reescalado. Se verifica despues de MMIN_DIM y puede forzar
    # una escala mayor. Por ejemplo, si esta establecido en 2 las imagenes se escalarán al doble
    # de ancho y alto, aun si MIN_DIM no lo requiere.
    # Sin embargo, en modo 'cuadrado', puede ser sobre pasado por MAX_DIM.
    ESCALA_MINIMA = 0
    # Numero de canales por color. RGB = 3, escala de grises = 1, RGB-D = 4
    CANALES = 3

    # Maximo numero de intancias (rerales) en una imagen
    MAX_R_INSTANCIAS = 100

    # Desviacion estandar para el refoinamiento de la caja contenedora para entrenamiento y deteccion
    ENT_CC_STD_DEV = numpy.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = numpy.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

################################################################################################
# Modelo Convolucional
################################################################################################

    # Celdas por lado de regilla
    # La imagen se divirá en una regilla de S x S celdas
    # Es importante para el calculo de las capas del perceptrón y el tensor de salida
    # Debe poder dividir los lados de la imagen de manera entera
    S = 4
    # Cantidad de Anchors (cajas predictivas), generas por cada celda de la regilla
    B = 5
    # Cantos Anchors se utilizarán para el entrenamiento en 1 imagen
    ANCHORS_ENTRENAMIENTO_IMAGEN = S*S*B
    
    # Tipo de Red
    # Tipo de red que es construida.
    #TODO: Describir tipos de red
    #"Conv24"
    #"Conv19"
    #"Res55"
    RED_TIPO = 'Conv19'
    
    # Tipo de salida:
    # "Y": Construye una salida con forma de Y, que tiene su disyunción al finalizar la red
    #   convolucional y clacula por separado, los Anchors y deltas
    # "L": Contruye una salida con forma de L, donde la salida que calcula las deltas, se
    #   se encuentra como una derivación de la salida que calcula los Anchors
    RED_TIPO_SALIDA = "Y"
    
    def __init__(self):
        """Set values of computed attributes."""
        # Tamaño de batch efectivo
        self.TAM_BATCH = self.IMG_GPU * self.GPUs
        # Tamaño de imagen de entrada
        if self.MODO_REESCALADO == "aleatorio":
            self.FORMA_IMAGEN = numpy.array([self.MIN_DIM, self.MIN_DIM,
                self.CANALES])
        else:
            self.FORMA_IMAGEN = numpy.array([self.MAX_DIM, self.MAX_DIM,
                self.CANALES])