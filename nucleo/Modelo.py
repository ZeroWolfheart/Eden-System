import random
import numpy
import logging
import math

import Utiles
from Dataset  import Dataset
from Configuracion import Configuracion

##############################################################################################
# Generacion de datos
##############################################################################################


def generador_Datos(dataset = Dataset, modo="entrenamiento", configuracion = Configuracion, revolver = True, tam_batch = 1):
    """
        Crea un generador de datos válido para Keras. Necesario para el entrenamiento de los
    modelos.
    
    # Argumentos:

        dataset: objeto de la clase Dataset. Inicializado y con valores en sus parametros
            entrenamiento y válidacion 
        
        modo: cualquiera de; "entrenamiento, validacion"
        
        configruacion: objeto de la clase o subclase configuracion
        
        revolver: booleano, indica si info_img debe ser ordenado aleatoriamente al inicio de
            cada epoca
        
        tam_batch: entero, indica el tamaño de batch a utilizar
        
    """
    assert modo in ['entrenamiento', 'validacion'], "Modo {}, no soportado".format(modo)
    
    if modo == 'entrenamiento':
        info_img = dataset.entrenamiento.copy()
    if modo == 'validacion':
        info_img = dataset.validacion.copy()
        
    index_batch = 0 # indice del elemento del batch
    index_imagen = -1
    contador_error = 0
    
    anclas, _centrosA, anclasR = Utiles.generar_Anchors_Celdas_V2(configuracion.ANCHOR_SCALAS,
                                                        forma_imagen=(configuracion.FORMA_IMAGEN[0],configuracion.FORMA_IMAGEN[1]),
                                                        S =configuracion.S,
                                                        B = configuracion.B)

    # Keras requiere un generador que corra indefinidamente
    while True:
        try:
            
            # Incrementar indice para seleccionar imagen
            index_imagen = (index_imagen + 1) % len(info_img)
            # Revolver al inicio de cada epoca
            if revolver and index_imagen == 0:
                random.shuffle(info_img)

            # Cargar imagen, mascara y cajas contenedoras
            imagen, _mascara, cajas_contenedoras, clases_imagen = cargar_Componentes_Imagen(dataset = dataset,
                                                                                           info_img = info_img,
                                                                                           indice = index_imagen,
                                                                                           configuracion = configuracion,
                                                                                           aumento = False)
            cajas_relativas = Utiles.convertir_Cajas_a_Relativas(cajas_contenedoras,
                                                                 (configuracion.FORMA_IMAGEN[0], configuracion.FORMA_IMAGEN[1]),
                                                                 configuracion.S)
            # Calcular IoU
            iou = Utiles.calcular_Sobreposiciones(anclas, cajas_contenedoras)
            
            # Calcular desviación entre caja contenedora y ancla
            identificador, cajasDelta, mejorCoincidencia = calcular_Deltas(anclas,
                                                                        cajas_contenedoras,
                                                                        iou,
                                                                        configuracion)
            if configuracion.RED_TIPO_SALIDA in ["L","Y"]:
                
                tensor1, tensor2 = codificar_tensores_Salida(S=configuracion.S,
                                                B=configuracion.B,
                                                C=configuracion.NUM_CLASES,
                                                anchors=anclasR,
                                                iou=iou,
                                                ids_Clase=clases_imagen,
                                                deltas = cajasDelta,
                                                identificador = identificador,
                                                mejor_Coincidencia=mejorCoincidencia,
                                                usar_Idf=configuracion.USAR_IDF)
                
            elif configuracion.RED_TIPO_SALIDA in ["I"]:
                                
                tensor1 = codificar_Unico_Tensor_Salida(S=configuracion.S,
                                                        B=configuracion.B,
                                                        C=configuracion.NUM_CLASES,
                                                        anchors=anclasR,
                                                        cajasR = cajas_relativas,
                                                        ids_Clase=clases_imagen,
                                                        identificador = identificador,
                                                        deltas=cajasDelta,
                                                        mejor_Coincidencia=mejorCoincidencia,
                                                        IoU = iou)
            # Inicializar  el array del batch
            if index_batch == 0:
                x = numpy.zeros(
                    (tam_batch,) + imagen.shape, dtype=numpy.float 
                )
                y = numpy.zeros(
                    (tam_batch,) + (configuracion.S, configuracion.S, configuracion.B, 5+configuracion.NUM_CLASES),
                    dtype=numpy.float
                )
                if configuracion.RED_TIPO_SALIDA in ["L","Y"]:
                    if configuracion.USAR_IDF:
                        tam_nodo_t2 = 5
                    else:
                        tam_nodo_t2 = 4
                    y2 = numpy.zeros(
                        (tam_batch,) + (configuracion.S, configuracion.S, configuracion.B, tam_nodo_t2),
                        dtype=numpy.float
                    )
            
            # Añadir elementos al batch
            x[index_batch] = imagen
            y[index_batch] = tensor1
            if configuracion.RED_TIPO_SALIDA in ["L","Y"]:
                y2[index_batch] = tensor2
            index_batch +=1
            
            if index_batch >= tam_batch:
                if configuracion.RED_TIPO_SALIDA in ["L","Y"]:
                    yield ({"img_entrada":x},{"tensor_salida":y, "tensor_salida2":y2})
                elif configuracion.RED_TIPO_SALIDA in ["I"]:
                    yield ({"img_entrada":x},{"tensor_salida":y})
                # Iniciar un nuevo batch
                index_batch=0
            
        except(GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Registrar fallo y saltar la imagen
            logging.exception("Error procesando la imagen {}".format(
                info_img[index_imagen]['rI']
            ))
            contador_error += 1
            if contador_error > 5:
                raise

def cargar_Componentes_Imagen(dataset = Dataset, info_img = [], indice = 0, configuracion = Configuracion, aumento = True):
    
    # Cargar imagen original
    imagen = dataset.cargar_Imagen(indice,info_img)
    # Cargar mascara y clases de las mismas
    mascara, clases_imagen = dataset.cargar_Mascara(indice, info_img)
    # Reescalar imagen
    imagen, _ventana, escala, relleno, aleatorio = Utiles.reescalar_Imagen(imagen,
                                                                          minDim=configuracion.MIN_DIM,
                                                                          maxDim=configuracion.MAX_DIM,
                                                                          minEscala=configuracion.ESCALA_MINIMA,
                                                                          modo='cuadrado')
    # Reescalar mascara
    mascara = Utiles.reescalar_Mascara(mascara, escala, relleno, aleatorio=aleatorio)
    
    # Seleccionar aleatoriamente y girar horizontalmente la imagen y mascara
    if aumento:
        if random.randint(0,1):
            imagen = numpy.fliplr(imagen)
            mascara = numpy.fliplr(mascara)
    
    # Extraer cajas contenedoras de la máscara
    cajas_contenedoras = Utiles.extraer_Cajas_Contenedoras(mascara)
    
    return imagen, mascara, cajas_contenedoras, clases_imagen

def calcular_Deltas(anchors, cajasContenedoras, iou, configuracion):
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
    
    # calcular sobreposiciones (calculado previamente con la función:
    # Utiles.calcular_Sobreposiciones)
    sobrePos = iou
    # Comparar Anchor y cajasContenedoras
    # Si un anchor se sobrepone a una caja contenedora con un IoU >= configuracion.DELTA_IOU_MIN_POSITIVO es positivo
    # Si por el contrario el valor de IoU < configuracion.DELTA_IOU_MAX_NEGATIVO es negativo
    # Los Anchors neutrales son aquellos que no cumplen con las condiciones anteriores
    # y no tienen influencia en la funcion de perdida.
    # Sin embargo, evitar dejar alguna caja sin clasifica.
    # Encambio, empatarla al Anchor mas cercano ( aun si su IuO es < 0.3)
    
    # 1. Identigficar y marcar Anchors negativos primero. Estos se sobreescibirán si
    # alguna caja contenedora empata con ellas.
    
    anchorIoUargmax = numpy.argmax(sobrePos, axis=1)
    anchorIoUmax = sobrePos[numpy.arange(sobrePos.shape[0]), anchorIoUargmax]
    identificador [anchorIoUmax < configuracion.DELTA_IOU_MAX_NEGATIVO] = -1
    
    # 2. Proponer un Anchor por cada  caja contenedora (independientemente de su valor IoU)
    # Si multiples Anchors tienen el mismo IoU, identificar todos ellos
    
    ioUargmax = numpy.argwhere(sobrePos == numpy.max(sobrePos, axis=0))[:, 0]
    identificador[ioUargmax] = 1
    
    # 3. Marcar los Anchor con alto IoU como positivos
    identificador[anchorIoUmax >= configuracion.DELTA_IOU_MIN_POSITIVO] = 1
    
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
    
    for i, a in zip(ids, anchors[ids]):
        # Anchor más cercano (IoU>=configuracion.DELTA_IOU_MIN_POSITIVO)
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
        cajasDelta[i] = [(gtCy - aCy)/aH,
                          (gtCx - aCx)/aW,
                          numpy.log(gtH/aH),
                          numpy.log(gtW/aW)]
        
    identificador =  ((identificador*5)+5)/10
    cajasDelta = (cajasDelta+10)/20
    return identificador, cajasDelta, anchorIoUargmax

##############################################################################################
# Formato de datos
##############################################################################################

def codificar_tensores_Salida(S=7, B=2, C=1, anchors=None, iou=None, ids_Clase=None,
                              deltas = None, identificador = None, mejor_Coincidencia = None,
                              usar_Idf = False):
    
    # Formato de los tensores
    # Tensor de Anchors
    tensor =  numpy.zeros(shape=(S,S,B,5+C))
    # Tensor de deltas
    if usar_Idf:
        tam_nodo_t2 = 5
    else:
        tam_nodo_t2 = 4
    tensor2 = numpy.zeros(shape=(S,S,B,tam_nodo_t2))
    # Contador de los tensores
    contador = 0
    # Llenar tensores con los datos
    for a in range(len(anchors)):
        j,i = int(anchors[a][0]), int(anchors[a][1])
        cy,cx = anchors[a][2], anchors[a][3]
        h,w = anchors[a][4], anchors[a][5]
        iou_anchor = max(iou[a])
        dCy, dCx, dH, dW = deltas[a]
        iden = identificador[a]
        
        if iden == 1:
            iou_anchor = 1.0
            tensor[j][i][contador][0]= cy
            tensor[j][i][contador][1]= cx
            tensor[j][i][contador][2]= h
            tensor[j][i][contador][3]= w
            tensor[j][i][contador][4]= iou_anchor
            tensor[j][i][contador][5+ids_Clase[mejor_Coincidencia[a]]] = 1
            
            tensor2[j][i][contador][0]=dCy
            tensor2[j][i][contador][1]=dCx
            tensor2[j][i][contador][2]=dH
            tensor2[j][i][contador][3]=dW
            if usar_Idf:
                tensor2[j][i][contador][4]=iden
        
        contador+=1
        if contador == B:
            contador=0
    return tensor, tensor2

def codificar_Unico_Tensor_Salida(S=7, B=2, C=1, anchors=None, cajasR = None,
                                  ids_Clase = None, identificador = None, 
                                  deltas = None, mejor_Coincidencia = None, IoU = None):

    # Tensor de Anchors
    tensor =  numpy.zeros(shape=(S,S,B,5+C))
    # Contador de los tensores
    contador = -1
    # print(IoU.shape)
    mejor_del_mejor=numpy.zeros((len(cajasR),1), dtype=numpy.int32)
    for i in range(len(cajasR)):
        mejor_IoU=0
        for j in range(len(mejor_Coincidencia)):
            if i==mejor_Coincidencia[j]:
                if IoU[j][i]>mejor_IoU:
                    mejor_IoU=IoU[j][i]
                    mejor_del_mejor[i]=j
                          
    # print(mejor_del_mejor)
    # Llenar tensores con los datos
    for a in range(len(anchors)):
        if a % (S*S)==0:
            contador+=1
        j,i = int(anchors[a][0]), int(anchors[a][1])
        # print(j,i)
        iden = identificador[a]
        if iden == 1.0 and a in mejor_del_mejor:
            tensor[j][i][contador][0] = cajasR[mejor_Coincidencia[a]][2] #cy
            tensor[j][i][contador][1] = cajasR[mejor_Coincidencia[a]][3] #cx
            tensor[j][i][contador][2] = cajasR[mejor_Coincidencia[a]][4] #h
            tensor[j][i][contador][3] = cajasR[mejor_Coincidencia[a]][5] #w
            tensor[j][i][contador][4] = 1.0 #pc
            tensor[j][i][contador][5+ids_Clase[mejor_Coincidencia[a]]] = 1
            # print(IoU[a][mejor_Coincidencia[a]])
        # else:
        #     tensor[j][i][contador][:4] = 10/20
    return tensor

##############################################################################################
# Decodificación de datos
##############################################################################################

def decodificar_Tensores(t1=None, t2=None, forma_Imagen=(0,0), S=7, B=2, usar_Idf = False):
    
    # Preparar listas de salida
    anchors_propuestos=[]
    clases_anchor=[]
    deltas_calculados=[]
    # Calcular altura y base de cada celda en pixeles
    altura_celda = forma_Imagen[0]//S
    base_celda = forma_Imagen[1]//S
    # Analizar la malla (S*S), para extraer las caracteristicas de los B Anchors
    # de cada celda, y las deltas de dichos Anchors
    for j in range(0,S):
        for i in range(0,S):
            for b in range(0,B):
                # Calcular distancia en pixeles, desde el borde de la imagen
                # hasta la celda que se analiza
                paso_x = i * base_celda
                paso_y = j * altura_celda
                # Extraer valores de los tensores
            
                # Tensor 1, Anchors, IoU y clases
                cy  = t1[j][i][b][0]
                cx  = t1[j][i][b][1]
                h   = t1[j][i][b][2]
                w   = t1[j][i][b][3]
                iou = t1[j][i][b][4]
                # Para todas las cajas, extraer probabilidad de clases
                probabilidad_clases = t1[j][i][b][5:]
                
                # Tensor 2, Deltas e Identificadores
                dy    = t2[j][i][b][0]
                dx    = t2[j][i][b][1]
                logdh = t2[j][i][b][2]
                logdw = t2[j][i][b][3]
                if usar_Idf:
                    idf = t2[j][i][b][4]
                
                # Codificar salida de T1: (cy,cx,h,w), relativos a la celda y las dimensiones
                # de la imagen a (y1,x1,y2,x2), coordenadas de las esquinas en pixeles
                # Centro de la caja:
                pcy = (cy*altura_celda)+ paso_y
                pcx = (cx*base_celda)+ paso_x
                # Altura y base de la caja:
                hh = h * forma_Imagen[0]
                ww = w * forma_Imagen[1]
                # Convertir a coordenadas de esquinas:
                x1 = pcx - (ww * 0.5)
                y1 = pcy - (hh * 0.5)
                x2 = x1 + ww
                y2 = y1 + hh
                # Regresar valores de delta a su escala Original
                # Deviación del centro
                dy = (dy*20)-10
                dx = (dx*20)-10
                # Diferencia de Altura y Base
                logdh = (logdh*20)-10
                logdw = (logdw*20)-10
                # Identificador de Anchor positivo
                if usar_Idf:
                    idf = ((idf*10)-5)/5
                
                # Integrar a las listas de salida
                anchors_propuestos.append([y1, x1, y2, x2, iou])
                clases_anchor.append(probabilidad_clases)
                if usar_Idf:
                    deltas_calculados.append([dy, dx, logdh, logdw, idf])
                else:
                    deltas_calculados.append([dy, dx, logdh, logdw])
    # Convertir a objeto numpy, para futuras operaciones
    anchors_propuestos = numpy.array(anchors_propuestos)
    clases_anchor = numpy.array(clases_anchor)
    deltas_calculados = numpy.array(deltas_calculados)
    
    # Regresar valores en numpy array
    return anchors_propuestos, deltas_calculados, clases_anchor


def decodificar_Unico_Tensor_Salida(t1=None, anchors=None, S=7, B=2, forma_imagen = (0,0)):
    # Preparar listas de salida
    cajas_propuestos=[]
    clases_caja=[]
    # Analizar la malla (S*S), para extraer las caracteristicas de los B Anchors
    # de cada celda, y las deltas de dichos Anchors
    for b in range(0,B):
        # Se inicia en esta forma, ya que los Anchor estan generados de tal forma
        # que el los anchor desde el 0 hasta el S*S pertencen a B[0] de la malla,
        # asi consecutivamente hasta que los (S*S) (n-1) pertenecen a los B[n-1] 
        # de cada celda
        for j in range(0,S):
            for i in range(0,S):
                # Extraer valores del tensor
                # Tensor 1, deltas y clases
                y  = t1[j][i][b][0]
                x  = t1[j][i][b][1]
                h  = t1[j][i][b][2]
                w  = t1[j][i][b][3]
                pc = t1[j][i][b][4]
                # Para todas las cajas, extraer probabilidad de clases
                probabilidad_clases = t1[j][i][b][5:]
                
                y = y*forma_imagen[0]
                x = x*forma_imagen[1]
                h = h*forma_imagen[0]
                w = w*forma_imagen[1]
                
                y1 = y-(h*0.5)
                x1 = x-(w*0.5)
                y2 = y1+h
                x2 = x1+w
                
                # # Extraer valores del tensor
                # # Tensor 1, deltas y clases
                # dy  = (t1[j][i][b][0]*20)-10
                # dx  = (t1[j][i][b][1]*20)-10
                # dh  = (t1[j][i][b][2]*20)-10
                # dw  = (t1[j][i][b][3]*20)-10
                # pc  = t1[j][i][b][4]
                # # Para todas las cajas, extraer probabilidad de clases
                # probabilidad_clases = t1[j][i][b][5:]
                
                # # Extraer información del anchor correspondiente al vector
                # # del tensor de salida en proceso
                # y1,x1,y2,x2 = anchors[contador]
                # contador+=1
                
                # # Calcular la caja correspondiente a partir de la desviación calculada
                # # y el ancla a la que corresponde
                # c_y1,c_x1,c_y2,c_x2 = Utiles.aplicar_Delta_Caja(caja=[y1,x1,y2,x2],delta=[dy,dx,dh,dw])
                
                # Añadir a las listas de salida
                # cajas_propuestos.append([c_y1,c_x1,c_y2,c_x2, pc])
                cajas_propuestos.append([y1,x1,y2,x2, pc])
                clases_caja.append(probabilidad_clases)

    # Convertir a objeto numpy, para futuras operaciones
    cajas_propuestos = numpy.array(cajas_propuestos)
    clases_caja = numpy.array(clases_caja)
    
    # Regresar valores en numpy array
    return cajas_propuestos, clases_caja