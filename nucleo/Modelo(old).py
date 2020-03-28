import os
import math
import random
import logging
import numpy
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# Librerias Keras
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
# Ficheros propios
import Utiles as uT

#############################################################################################
# FUNCIONES DE UTILIDAD Y CONSTRUCCION                                                      #
#############################################################################################

def computarFormasBackbone(configuracion):
    """ Calcula el alto y base de cada capa de la red Neuronal Convolucional (Backbone),
    segun la configuración provista por el objeto de entrada.
    
    # Argumentos:

        configuracion: objeto de configuracion
        
    # Salidas
        [[altura, base],...] :  Numpy array
    """
    formaImg = configuracion.FORMA_IMAGEN
    h = formaImg[0]
    w = formaImg[1]
    temp = []
    for stride in configuracion.BACKBONE_STRIDES:
        h = int(math.ceil(h/stride))
        w = int(math.ceil(w/stride))
        temp.append([h,w])
    formaBack = numpy.array(temp, dtype=int)
    return formaBack

#############################################################################################
# GENERACION DE DATOS                                                                       #
#############################################################################################

def cargarImagenComp (dataSet, modo, configuracion, idImagen, aumento = False, usarMiniMascara = False):
    """ Carga y retorna información sobre una imagen (imagen, mascara, cajas contenedoras)
    
    # Argumentos:

        dataSet: objeto tipo Dataset
        modo: cadena que puede ser "entrenamiento" o "validacion"
        configuracion: objeto tipo Configuracion
        idImagen: entero, id de una imagen del conjunto de validacion o entrenamiento del Dataset
        aumento: selecciona al azar 1 imagen y la pone en modo espejo
        usarMiniMascara: minimiza la mascara, con los parametros dados en Configuracion, con el fin de ahorar memporia
    
    # Salidas:

        imagen: Arreglo de matrices que  contienen los 3 canales de una imagen, reescalados sengun la configuracion
        mataImagen: Arreglo de 1 dimension que contiene  informacion de la imagen (ver componerMetaDataImg())
        idsClaseImagen: Arreglo con los id de las clases que se encuentran en la imagen
        cajasContenedoras: arreglo de (x1, y1, x2, y2) de cada una de las cajas calculadas a partir de la mascara
        mascara: arreglo de matrices que identifican el area ocupada por un objeto de alguna clase
    """
    assert modo in ['entrenamiento', 'validacion']
    if modo == 'entrenamiento':
        conjunto = dataSet.entrenamiento
    if modo == 'validacion':
        conjunto = dataSet.validacion
    # Cargar imagen
    imagen = dataSet.cargarImagen(idImagen,conjunto)
    # Cargar mascara y arreglo de clases
    mascara, idsClaseImagen = dataSet.cargarMascara(idImagen, conjunto)
    # Extraer la forma original de la imagen
    formaOriginal = imagen.shape
    # Reescalar imagen
    imagen, ventana, escala, relleno, aleatorio = uT.reescalarImagen(
        imagen, minDim=configuracion.MIN_DIM, maxDim=configuracion.MAX_DIM,
        minEscala= configuracion.ESCALA_MINIMA, modo=configuracion.MODO_REESCALADO
    )
    # Reescalar mascara
    mascara = uT.reescalarMascara(
        mascara, escala, relleno, aleatorio=aleatorio
    )
    
    # Seleccionar aleatoriamente y girar horizontalmente 1 imagen
    if aumento:
        if random.randint(0,1):
            imagen = numpy.fliplr(imagen)
            mascara = numpy.fliplr(mascara)
    
    # Algunas cajas pueden ser todo 0, si la máscara que corresponde fue cortada por la funcion aleatoria
    # en este punto se filtran esas cajas
    _idx = numpy.sum(mascara, axis=(0,1)) > 0
    mascara = mascara [:,:, _idx]
    idsClaseImagen = idsClaseImagen[_idx]
    # Extraer cajas contenedoras
    # cajasContenedoras: [ssss,(y1, x1, y2, x2)]
    cajasContenedoras = uT.extraerCajasContenedoras(mascara)
    
    # Reescalar máscara a un tamaño más pequeño para reducir el uso de memoria
    if usarMiniMascara:
        mascara = uT.minimzarMascara(cajasContenedoras, mascara, configuracion.MINIMASCARA_SHAPE)
    
    # Construir metadata de imagen
    mataImagen = componerMetaDataImg(idImagen, formaOriginal, imagen.shape, ventana, escala)
    
    return imagen, mataImagen, idsClaseImagen, cajasContenedoras, mascara

def calcularDeltas(formaImagen, anchors, idsClases, cajasContenedoras, configuracion):
    """ Dados los Anchors y las cajas conteneodras, se computa su sobreposicion y se indentifica
        los Anchors positivos y sus deltas para refinar su posicion y que empaten con la caja contenedora
        correspondiente
    # Argumentos:
    # Salida:
    """
    
    # identificador : 1 = Anchor positivo, -1 = Anchor negativo, 0 = neutral
    identificador = numpy.zeros([anchors.shape[0]], dtype=numpy.int32)
    # cajas delta: [(dy, dx, log(dh), log(dw))]
    cajasDelta = numpy.zeros((configuracion.ANCHORS_ENTRENAMIENTO_IMAGEN, 4))
    
    # calcular sobreposiciones
    sobrePos = uT.calcularSobreposiciones(anchors, cajasContenedoras)
    
    # Comparar Anchor y cajasContenedoras
    # Si un anchor se sobrepone a una caja contenedora con un IoU >= 0.7 es positivo
    # Si por el contrario el valor de IoU < 0.3 es negativo
    # Los Anchors neutrales son aquellos que no cumplen con las condiciones anteriores
    # y no tienen influencia en la funcion de perdida.
    # Sin embargo, evitar dejar alguna caja sin clasifica.
    # Encambio, empatarla al Anchor mas cercano ( aun si su IuO es < 0.3)
    
    # 1. Identigficar y marcar Anchors negativos primero. Estos se sobreescibirán si
    # alguna caja contenedora empata con ellas.
    
    anchorIoUargmax = numpy.argmax(sobrePos, axis=1)
    anchorIoUmax = sobrePos[numpy.arange(sobrePos.shape[0]), anchorIoUargmax]
    identificador [anchorIoUmax < 0.3] = -1
    
    # 2. Proponer un Anchor por cada  caja contenedora (independientemente de su valor IoU)
    # Si multiples Anchors tienen el mismo IoU, identificar todos ellos
    
    ioUargmax = numpy.argwhere(sobrePos == numpy.max(sobrePos, axis=0))[:, 0]
    identificador[ioUargmax] = 1
    
    # 3. Marcar los Anchor con alto IoU como positivos
    identificador[anchorIoUmax >= 0.7] = 1
    
    # Submuestrear para balancear Anchors negativos y positivos
    # No dejar que los positivos sean más de la mitad
    
    ids = numpy.where(identificador == 1)[0]
    extra = len(ids) - (configuracion.ANCHORS_ENTRENAMIENTO_IMAGEN // 2)
    if extra > 0:
        # Reiniciar los extras a neutral(0)
        ids = numpy.random.choice(ids, extra, replace=False)
        identificador[ids] = 0
    
    # Lo mismo para los Anchor negativos
    ids = numpy.where(identificador == -1)[0]
    extra = len(ids) - (configuracion.ANCHORS_ENTRENAMIENTO_IMAGEN -
                        numpy.sum(identificador == 1))
    if extra > 0:
        # Reiniciar los extras a neutral
        ids = numpy.random.choice(ids, extra, replace=False)
        identificador[ids] = 0
    
    # Para los Anchors positivos, calcular la forma y la escala necesaria para
    # transformalos en las cajas contenedoras correspondientes
    ids = numpy.where(identificador == 1)[0]
    ix = 0 # index en cajasDelta
    
    for i, a in zip(ids, anchors[ids]):
        # Anchor más cercano (IoU>=0.7)
        gt = cajasContenedoras[anchorIoUargmax[i]]
        
        # Convertirt coordenadas a centro más base/altura
        # Caja
        gtH = gt[2] - gt[0]
        gtW = gt[3] - gt[1]
        gtCy = gt[0] + 0.5 * gtH
        gtCx = gt[1] + 0.5 * gtW
        # Anchor
        aH = a[2] - a[0]
        aW = a[3] - a[1]
        aCy = a[0] + 0.5 * aH
        aCx = a[1] + 0.5 * aW
        
        # Calcular el refinamiento que el modelo debe predecir
        cajasDelta[ix] = [(gtCy - aCy)/aH,
                          (gtCx - aCx)/aW,
                          numpy.log(gtH/aH),
                          numpy.log(gtW/aW)]
        # Normalizar
        cajasDelta[ix] /= configuracion.ENT_CC_STD_DEV
        ix+=1
    return identificador, cajasDelta
    
def generadorDatos(dataSet, modo, configuracion, revolver = True, tamBach =1 ):
    # Válidar si el modo es entrenamiento o validación, para el generador
    assert modo in ['entrenamiento', 'validacion']
    b = 0
    indexImagen = -1
    if modo == 'entrenamiento':
        idsImagen = dataSet.entrenamiento[:]
    if modo == 'validacion':
        idsImagen = dataSet.validacion[:]
    conteoError = 0
    
    # Dimensiones de cada capa de la red
    formasBack = computarFormasBackbone(configuracion)
    
    # Anclas (Anchors): cajas predictivas (posibles ocurrencias de objeto, se tomaran las mejores)
    cajasPredictivas = uT.generarAnchorsCapa(configuracion.ANCHOR_SCALAS,
                                             configuracion.ANCHOR_FACTORES,
                                             formasBack,
                                             configuracion.BACKBONE_STRIDES,
                                             configuracion.ANCHOR_STRIDE)
    
    #Keras requiere un generador que corra indefinidamente
    while True:
        try:
            # Incrementar el indice para tomar la siguiente imagen
            indexImagen = (indexImagen + 1)  % len(idsImagen)
            # Revolver imagenes si se esta al inicio de la epoca
            if revolver and indexImagen == 0:
                random.shuffle(idsImagen)
            
            # Obtener máscara, clases e imagen
            imagen, metaImagen, idsClaseImagen, cajas, mascara = cargarImagenComp (dataSet, modo, configuracion, indexImagen)

            # Deltas
            match, deltas = calcularDeltas(imagen.shape, cajasPredictivas, idsClaseImagen, cajas, configuracion)
            
            # Inicializar array del batch
            if b == 0:
                bMeta   = numpy.zeros(
                    (tamBach,) + metaImagen.shape, dtype = metaImagen.dtype)
                bMatch  = numpy.zeros(
                    [tamBach, cajasPredictivas.shape[0], 1], dtype = match.dtype)
                bDeltas = numpy.zeros(
                    [tamBach, configuracion.ANCHORS_ENTRENAMIENTO_IMAGEN, 4], dtype=deltas.dtype)
                bImagen = numpy.zeros(
                    (tamBach,) + imagen.shape, dtype = numpy.float32)
                bClasId = numpy.zeros(
                    (tamBach, configuracion.MAX_R_INSTANCIAS), dtype=numpy.int32)
                bCConte = numpy.zeros(
                    (tamBach, configuracion.MAX_R_INSTANCIAS, 4), dtype = numpy.int32)
                bMascar = numpy.zeros(
                    (tamBach, mascara.shape[0], mascara.shape[1],
                    configuracion.MAX_R_INSTANCIAS), dtype = mascara.dtype)
                
            # Si hay mas instancias de las que caben en el arreglo, hacer sub muestreo
            if cajas.shape[0] > configuracion.MAX_R_INSTANCIAS:
                ids = numpy.random.choice(
                    numpy.arange(cajas.shape[0]), configuracion.MAX_R_INSTANCIAS, replace = False)
                idsClaseImagen = idsClaseImagen[ids]
                cajas = cajas[ids]
                mascara = mascara[:,:,ids]
            
            # Añadir al batch
            bMeta[b] =metaImagen
            bMatch[b] =match[:, numpy.newaxis]
            bDeltas[b] =deltas
            bImagen[b] = mediarImagen(imagen.astype(numpy.float32), configuracion)
            bClasId[b, :idsClaseImagen.shape[0]] = idsClaseImagen
            bCConte[b, :cajas[0]] = cajas
            bMascar[b, :, :, :mascara.shape[-1]] = mascara
            
            b += 1
            
            # Batch lleno
            if b >= tamBach:
                entradas = [bImagen, bMeta, bMeta, bDeltas, bClasId, bCConte, bMascar]
                salidas =[]
                yield entradas, salidas
                # Iniciar un nievo batch
                b = 0
        except(GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # log y saltar imagen
            logging.exception("Error procesando imagen {}".format(
                dataSet.getDireccionImagen(indexImagen, idsImagen)
            ))
            conteoError += 1
            if conteoError > 5:
                raise    

#############################################################################################
# FORMATO DE DATOS                                                                          #
#############################################################################################

def componerMetaDataImg(idImagen, formaOriginal, forma, ventana, escala):
    """ Toma los atributos de una imagen y los coloca en un arreglo 1D.
    
    # Argumentos:
        
        idImagen: id de una imagen, util para el debuggin
        formaOriginal: [H, W, C] antes de las transformaciones de reescalado
        forma: [H, W, C] despues de las transformaciones de reescalado
        ventana: (y1, x1, y2, x2) en pixeles. Area de la imagen donde se encuentra el grafico original, sin el relleno
        escala: factor de escala apliacado a la imagen original (float32)
    
    # Salida:

        meta  : arreglo de 1 dimension con la información anterior
    """
    
    meta = numpy.array(
        [idImagen] +            # 1
        list(formaOriginal) +   # 3
        list(forma) +           # 3
        list(ventana) +         # 4
        [escala]                # 1
    )
    return meta

def mediarImagen(imagen, configuracion):
    """ Espera una imagen RGB (o un array de imagenes) y extrae
    el pixel medio y lo convierte en flotante. Requiere una imagen
    con los colores en el dorde RGB
    """
    return imagen.astype(numpy.float32) - configuracion.MEDIA_PIXEL

#############################################################################################
# CLASE MODELO                                                                              #
#############################################################################################

class Modelo:
    """ Encapsula la funcionalidad de la red Neuronal convolucional(backbone)
        Utiliza las herramientas de Keras
    """
    
    def __init__(self, modo, configuracion, directorio):
        """
            modo: cualquiera "entrenamiento" o "inferencia"
            configuracion: una instancia de la clase o subclase de Configuracion
            directorio: ubicacion deonde se guardan los registros, modelo y pesos generados
        """
        assert modo in ['entrenamiento' , 'inferencia']
        self.modo = modo
        self.config = configuracion
        self.dirr = directorio
        self.setLogDir()
        self.modeloKeras = self.construir(modo=modo, configuracion = configuracion)
        
    def construir(modo, configuracion):
        """ Define al arquitectura de la Red Neuronal
        
            modo: cualquiera "entrenamiento" o "inferencia"
        """
        assert modo in ['entrenamiento' , 'inferencia']
        
        # La imagendebe ser dividisible entre 2 multiples veces
        h,w = configuracion.FORMA_IMAGEN[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("El tamaño de imagen debe ser divisible entre 2 al menos 6 veces"
                            "de forma entera, con el fin de evitar fracciones durante el reescalado."
                            "Por ejemplo: 256, 320, 384, 448, 512, ... etc. ")

        # Entradas
        entradaImagen = KL.Input(
            shape=[None, None, configuracion.FORMA_IMAGEN[2]], name= "entrada_Imagen"
        )
        entradaMetaImagen = KL.Input(
            shape=[configuracion.IMAGEN_META_TAM], name= "entrada_Meta"
        )
        
        if modo == "entrenamiento":
            # Deltas
            entradaMatch = KL.Input(
                shape=[None,1], name="entrada_Match", dtype= tf.int32
            )
            entradaDeltas = KL.Input(
                shape=[None, 4], name="entrada_Delta", dtype= tf.float32
            )
            # Elementos de deteccion
            # 1. Clases de la imagen
            entradaIdsClaseImg = KL.Input(
                shape=[None], name="entrada_ids_clase", dtype=tf.int32
            )
            # 2. Cajas contenedoras en pixeles
            # [batch, MAX_R_INSTANCIAS, (y1, x1, y2, x2)] en coordenadas
            entradaCajas = KL.Input(
                shape=[None,4], name="entrada_cajas", dtype=tf.float32
            )
            # Normalizar coordenadas
            cajas = KL.Lambda(lambda x: normalizarCajas(
                x, K.shape(entradaImagen)[1:3]))(entradaCajas)
            # 3. Mascara
            # [batch, h, w, MAX_R_INSTANCIAS]
            if configuracion.USAR_MINIMASCARA:
                entradaMascara = KL.Input(
                    shape=[configuracion.MINIMASCARA_SHAPE[0],
                           configuracion.MINIMASCARA_SHAPE[1], None],
                    name="entrada_mascara", dtype=bool
                )
            else:
                entradaMascara = KL.Input(
                    shape=[configuracion.FORMA_IMAGEN[0],
                           configuracion.FORMA_IMAGEN[1], None],
                    name="entrada_mascara", dtype=bool
                )
        elif modo == "inferencia":
            entradaAnchors = KL.Input(
                shape=[None, 4], name = "entrada_anchors"
            )
        
        
    def entrenar (self, dataSet, umbralAprendizaje, epocas, augmentation=None):
        
        # Generadores de datos
        generadorEntrenamiento = generadorDatos()
        generadorValidacion = generadorDatos()
        return generadorValidacion, generadorEntrenamiento