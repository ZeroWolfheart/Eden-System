import numpy
import random
import scipy
import math
import skimage.transform
import warnings
from distutils.version import LooseVersion

#######################################################################################################
# Manipulación de tamaño y forma de imagen y máscaras
#######################################################################################################

def reescalar (imagen, formaSalida, orden = 1, modo = 'constant', cVal = 0, clip = True,
               preservarRango = False, antiAliasing = False, antiAliasingSigma = None):
    """ Contenedor para Scikit-Image resize()
        
        Scikit-Image genera advertencias cada vez que se llama a resize() si no recibe los parametros correctos.
        Estos dependen de la versión de skimage. Para resolver esto se usan diferentes parametros por version
        y se provee de un control central de esta función.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # Nuevo en 0.14: anti_aliasing.
        return skimage.transform.resize(
            imagen, formaSalida, order=orden, mode=modo, cval=cVal, clip=clip,
            preserve_range=preservarRango, anti_aliasing=antiAliasing,
            anti_aliasing_sigma=antiAliasingSigma)
    else:
        return skimage.transform.resize(
            imagen, formaSalida,
            order=orden, mode=modo, cval=cVal, clip=clip,
            preserve_range=preservarRango)

def reescalar_Imagen(imagen, minDim=None, maxDim=None, minEscala=None, modo = "cuadrado"):
    """ Rescala una imagen, manteniendio su aspecto.
    
    # Argumentos:
    
    minDim: si se provee, reescala la imagen de tal forma que, la dimensión minima de la imagen == minDim
    maxDim: si se provee, asegura que el lado más grande de la imagen no exceda maxDim
    minEscala: si se provee, asegura que la imagen sera escalada por lo menos a este procentaje, incluso si es menor que minDim
    modo: modo de reescalado
    
        none: regresa la imagen sin cambios
        cuadrado: Reescala y rellena con ceros para tener una imagen de maxDim * maxDim
        relleno64: rellena base y altura con zeros, para hacerlos multiplos de 64. Sí minDim o minEscala son provistos, escala la imagen antes de rellenar. maxDim es ignorado en este modo.
        aleatorio: toma fragmentos aleatorios de la imagen. Primero escala la imagen basado en minDim y minEscala, despues, toma un fragmento aleatorio de tamaño minDim * minDim
    
    # Devuelve:
        imagen: imagen reescalada
        ventana: (y1, x1, y2, x2). Si se prevee maxDim, la imagen deberia tener relleno. De ser asi
            la ventana son las coordenadas de la parte de la imagen que contiene la imagen original
        escala: factor utilizado para reescalar la imagen
        relleno: relleno añadido a la imagen [(superio, inferior), (izquierda, derecha), (0, 0)]
    """
    # Almacenar dtype de la imagen, para regresarla de la misma forma
    dtypeImagen = imagen.dtype
    
    # valores de retorno por default
    h, w = imagen.shape[:2]
    ventana = (0, 0, h, w)
    escala  = 1
    relleno = [(0,0), (0,0), (0,0)]
    aleatorio = None
    
    if modo == "none":
        return imagen, ventana, escala, relleno, aleatorio
    
    # Escala
    if minDim:
        escala = max(1, minDim / min(h,w))
    if minEscala and escala < minEscala:
        escala = minEscala
    
    # Excede la dimensión maxima? (maxDim)
    if maxDim and modo == "cuadrado":
        maxImag = max (h,w)
        if round(maxImag*escala)>maxDim:
            escala = maxDim/maxImag
    
    # Reescalar imagen usando interpolación bilinear
    if escala != 1:
        imagen = reescalar(imagen, (round(h*escala), round(w*escala)), preservarRango = True)

    # Necesita relleno o muestreo aleatorio?
    if modo == "cuadrado":
        # Obtener las nuevas dimensiones
        h,w = imagen.shape[:2]
        limSup = (maxDim - h)//2
        limInf = maxDim - h - limSup
        limIzq = (maxDim - w)//2
        limDer = maxDim - w - limIzq
        # Delimitar área de imagen
        relleno = [(limSup, limInf), (limIzq, limDer), (0,0)]
        # Rellenar
        imagen = numpy.pad(imagen, relleno, mode='constant',constant_values=0)
        # Obtener ventana correspondiente a la imagen original
        ventana = (limSup, limIzq, h + limSup, w + limIzq)
    elif modo == "relleno64":
        h,w = imagen.shape[:2]
        # Ambas dimensiones deben ser divisibles entre 64
        assert minDim % 64  == 0, "La dimensión minima debe ser divisible entre 64"
        # Altura
        if h % 64 > 0:
            maxH = h - (h % 64) + 64
            limSup =(maxH-h)//2
            limInf = maxH - h - limSup
        else:
            limSup = limInf = 0
        # Base (anchura)
        if w % 64 > 0:
            maxW = w -(w % 64) + 64
            limIzq = (maxW-w)//2
            limDer = maxW - w - limIzq
        else:
            limIzq = limDer = 0
        
        relleno = [(limSup, limInf), (limIzq, limDer), (0,0)]
        imagen = numpy.pad(imagen, relleno, mode = 'constant', cosntant_values = 0)
        ventana = (limSup, limIzq, h + limSup, w + limIzq)
    
    elif modo == "aleatorio":
        # Tomar un fragmento aleatorio
        h, w = imagen.shape[:2]
        y = random.randint(0, h-minDim)
        x = random.randint(0, w-minDim)
        aleatorio = (y, x, minDim, minDim)
        imagen = imagen[y:y + minDim, x:x + minDim]
        ventana = (0 , 0, minDim, minDim)
    
    else:
        raise Exception("Modo {}, no soportado".format(modo))
    return imagen.astype(dtypeImagen), ventana, escala, relleno, aleatorio

def reescalar_Mascara (mascara, escala, relleno, aleatorio = None):
    """ Reescala una mascara, usando la escala y relleno especificados
    Normalmente se utiliza la mascara y relleno obtenidos de reescalarImagen() para asegurar
    la consistencia entre la imagen y su mascara despues del reescalado.
    
    # Argumentos:
    
        escala: factor de escalado
        relleno: relleno a agregar a la mascara en formato [(superior, inferior), (izquierda, derecha), (0,0)]
        aleatorio: si existe, especifica el fragmento extraido de la imagen en formato (y, x, minDim, minDim)
    
    # Devuelve:
        
        mascara: mascara reescalada
    """
    # Suprimir advetencias de scipy 0.13.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mascara = scipy.ndimage.zoom(mascara, zoom=[escala, escala, 1], order=0)
    if aleatorio is not None:
        y, x, h, w = aleatorio
        mascara = mascara[y:y + h, x:x + w]
    else:
        mascara = numpy.pad(mascara, relleno, mode='constant', constant_values=0)
    return mascara

def minimzar_Mascara (cajasContenedoras, mascara, forma):
    """ Reescala la mascara a la forma minima para reducir uso de memoria.

        Las miniMascaras pueden ser reescalas de vuelta a la escala de la imagen con expandirMascara()
    """
    miniMascara = numpy.zeros(forma + (mascara.shape[-1],), dtype=bool)
    for i in range(mascara.shape[-1]):
        # Castear la mascara
        m = mascara[:,:,i].astype(bool)
        y1, x1, y2, x2 = cajasContenedoras[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Caja contenedora inválida, con area de 0")
        # Reescalar usando la interpolacion bilinear
        m = reescalar(m, forma)
        miniMascara[:,:,i] = numpy.around(m).astype(numpy.bool)
    
    return miniMascara

def expandir_Mascara(cajasContenedoras, miniMascara, formaImagen):
    """ Reescala miniMascaras al tamaño de la imagen. revierte el cambio de minimzarMascara().

    """
    mascara = numpy.zeros(formaImagen[:2] + (miniMascara.shape[-1],), dtype=bool)
    for i in range(mascara.shape[-1]):
        m = miniMascara[:, :, i]
        y1, x1, y2, x2 = cajasContenedoras[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Reescalar con interpolacion bilinear
        m = reescalar(m, (h, w))
        mascara[y1:y2, x1:x2, i] = numpy.around(m).astype(numpy.bool)
    return mascara

#######################################################################################################
# Manipulación de cajas Contenedoras
#######################################################################################################

def extraer_Cajas_Contenedoras(mascara):
    """ Calcula las cajas contenedoras de la máscara de entrada
    
    # Argumentos:
        
        mascara de entrada: [altura, base, canal]. La mascara debe ser una matriz binaria
        
    # Salida:

        cajas_Contenedoras: [(y1, x1, y2, x2),...] donde
            y1,x1,y2,x2 son coordenadas en pixeles de las esquinas
            que delimitan la caja
            Todos los valores son enteros
    """
    cajas = numpy.zeros([mascara.shape[-1],4], dtype=numpy.int32)
    for i in range(mascara.shape[-1]):
        m = mascara[:,:, i]
        # Caja contenedora
        indicesHorizontales = numpy.where(numpy.any(m, axis=0))[0]
        indicesVerticales = numpy.where(numpy.any(m, axis=1))[0]
        
        if indicesHorizontales.shape[0]:
            x1, x2 = indicesHorizontales[[0,-1]]
            y1, y2 = indicesVerticales[[0,-1]]
            # x2 e y2 no son parte de la caja
            x2 += 1
            y2 += 1
        else:
            # en caso que no existan elementos en ese canal de máscara
            x1, x2, y1, y2 = 0, 0, 0, 0
        cajas[i] = numpy.array([y1, x1, y2, x2])
    return cajas.astype(numpy.int32)

def convertir_Cajas_a_Relativas(cajas, forma_imagen =(0,0), S=7):
    """ Dado un vector con las coordenadas de las esquinas de la caja,
    devuelve la información de estas de forma relativa al tamño de la imagen
    y su pocisión dentro de la celda que lo contiene.
    
    # Argumentos:
    
        cajas: arreglo numpy con formato [[y1,x1,y2,x2],...], que contiene las
            coordenadas en pixeles de las cajas de una imagen
            (calculadas de su máscara)
        
        forma_imagen: lista con formato (altura,base) de la imagen en pixeles
        
        S: numero entero; cantidad de celdas en las que se divide la imagen, tanto
            de manera vertical, como horizontal para crear una malla de SxS
    
    # Salida:
    
        cajasR: arreglo numpy con formato [[y,x,cy,cx,h,w],...] donde
            y,x representan la posicion de la celda que contiene el centro de la caja
            cy,cx, son la posicion relativa, del centro de la caja dentro de su celda
            h,w son la altura y la base de la caja con respecto a la imagen
            
    """
    # Crear array de cajas
    cajasR = numpy.zeros((cajas.shape[0], 6), dtype=numpy.float)
    # Obtener base y altura de la imagen
    altura_celda = forma_imagen[0]/S
    base_celda = forma_imagen[1]/S
    
    for i in range(len(cajas)):
        # Obtener coordenadas de esquinas
        y1, x1, y2, x2 = cajas[i][0],cajas[i][1], cajas[i][2], cajas[i][3]
        # Calcular la altura de la caja en pixeles
        h = y2-y1
        w = x2-x1
        # Calcular el centro de la  en pixeles
        cx = int(w/2) + x1
        cy = int(h/2) + y1
        # Calcular la altura y base relativa de la caja con respecto a la imagen
        hr = h/forma_imagen[0]
        wr = w/forma_imagen[1]
        # Identificiar en que celda de la malla se ubica el centro de la caja
        celdax = int(cx // base_celda)
        celday = int(cy // altura_celda)
        # Calcular la posicion relativa del centro dentro de la celda que lo contiene
        cxr = (cx-(base_celda*celdax))/base_celda
        cyr = (cy-(altura_celda*celday))/altura_celda
        # Añadir al vector, la información de la caja:
        # Coordenadas de la celda, posición ralativa del centro de la caja, con respecto
        # a la celda y altura y base de la caja, con respecto a la imagen
        cajasR[i] = numpy.array([celday, celdax, cyr, cxr, hr,wr])
    return cajasR.astype(numpy.float)

def calcular_IoU (caja, cajas, areaCaja, areaCajas):
    """ Calcula el IoU entre la caja, y el arreglo de cajas
    
    # Argumentos:
    
        caja: vector 1D [y1, x1, y2, x2]
        cajas: [(y1, x1, y2, x2), ....]
        areaCaja: area de la caja. float
        areaCajas: array de las areas de las cajas del vector cajas
        
    """
    # Calcular intersección entre areas
    y1 = numpy.maximum(caja[0], cajas[:, 0])
    y2 = numpy.minimum(caja[2], cajas[:, 2])
    x1 = numpy.maximum(caja[1], cajas[:, 1])
    x2 = numpy.minimum(caja[3], cajas[:, 3])
    
    interseccion = numpy.maximum(x2 - x1, 0) * numpy.maximum(y2 - y1, 0)
    union = areaCaja + areaCajas[:] - interseccion[:]
    iou = interseccion / union
    
    return iou

def calcular_Sobreposiciones(cajas1, cajas2):
    """ Calcula la Interseccion sobre la Union (IoU) emtre los dos juegos de cajas
        
    # Argumentos:
    
        cajas1, cajas2: [[y1, x1, y2, x2],...], para mejor rendimiento, cajas 1 es el set más largo
    """
    # Areas de los  Anchors y cajas contenedoras
    area1 = (cajas1[:,2]- cajas1[:,0]) * (cajas1[:,3]- cajas1[:,1])
    area2 = (cajas2[:,2]- cajas2[:,0]) * (cajas2[:,3]- cajas2[:,1])
    
    # Calcular sobreposicion y generar matriz [ cajas1 contador, cajas2 contador]
    # Cada celda contiene el valor IoU
    sobreposiciones = numpy.zeros((cajas1.shape[0], cajas2.shape[0]))
    for i in range(sobreposiciones.shape[1]):
        caja2 = cajas2[i]
        sobreposiciones[:,i] = calcular_IoU(caja2, cajas1, area2[i], area1)
    return sobreposiciones

#######################################################################################################
# Anchors (Cajas predictivas, propuestas)
#######################################################################################################

def generar_Anchors_Aleatorio(escalas, factores, dimensiones_imagen=(0,0), S=7, B=2):
    """
    # Argumentos:
    
        B: cantidad de Anchors(cajas contenedoras) a generar por celda
        S: cantidad de celdas a lo largo y ancho de la imagen, para crear una
            malla de SxS
        dimensiones_imagen: lista, con [alto, base] de la imagen para la que se crearán
            las cajas contenedoras
        escalas: lista con la dimension de 1 lado de los Anchor a calcular. Ejemplo [31,  64, 128]
        factores: lsita con los factores de tamaño de los Anchors. Ejemplo [0.5, 1 ,2]
    """
    paso_y = dimensiones_imagen[0]/S
    paso_x = dimensiones_imagen[1]/S
    anchors =[]
    centros =[]
    anchorsR = []
    for j in range(0,S):
        for i in range(0,S):
            # Espacio en pixeles, previo a la celda que se computa actualmente
            avance_y = paso_y * j
            avance_x = paso_x * i
                        
            # Calcular B cajas para cada celda
            for _b in range(0,B):
                # Calcular factores de desplazamiento sobre los ejes
                # para el centro de la caja dentro de la celda
                if j == 0:
                    fcy = random.uniform(0.5,0.99)
                elif j == (S-1):
                    fcy = random.uniform(0.01,0.49)
                else:
                    fcy = random.uniform(0.01,0.99)
                if i == 0:
                    fcx = random.uniform(0.5,0.99)
                elif i == (S-1):
                    fcx = random.uniform(0.01,0.49)
                else:
                    fcx = random.uniform(0.01,0.99)
                # Calcular el centro en pixeles
                cy = int((paso_y*fcy) + avance_y)
                cx = int((paso_x*fcx) + avance_x)
                    
                # Calcular base y altura
                h = int(escalas[random.randint(0,len(escalas)-1)] * factores[random.randint(0,len(factores)-1)])
                w = int(escalas[random.randint(0,len(escalas)-1)] * factores[random.randint(0,len(factores)-1)])
                # Calcular coordenadas de puntos
                y1 = int(cy - 0.5 * h)
                y2 = int(cy + 0.5 * h)
                x1 = int(cx - 0.5 * w)
                x2 = int(cx + 0.5 * w)
                # La caja es válida?
                # De no serlo, se corrigen sus dimensiones sin alterar
                # la posición del centro
                if y1 < 0:
                    dif = y1
                    y1 = 2
                    y2+= dif
                if x1 < 0:
                    dif = x1
                    x1 = 2
                    x2+= dif
                if y2 > dimensiones_imagen[0]:
                    dif = y2 - dimensiones_imagen[0]
                    y2 = dimensiones_imagen[0]-2
                    y1+= dif
                if x2 > dimensiones_imagen[1]:
                    dif = x2 - dimensiones_imagen[1]
                    x2 = dimensiones_imagen[1]-2
                    x1+= dif                
                # Añadir elementos a su respectivo arreglo
                anchors.append([y1,x1,y2,x2])
                centros.append([cy,cx])

    # Normalizar salidas como objeto numpy
    anchors = numpy.array(anchors,dtype=numpy.int32)
    centros = numpy.array(centros, dtype=numpy.int32)
    # Calcular valores relativos
    anchorsR = convertir_Cajas_a_Relativas(anchors,forma_imagen=dimensiones_imagen,S=S)
    return anchors, centros, anchorsR


def generar_Anchors_Celdas(escalas = [32, 64, 128, 256],
                           factores = [0.5, 1, 2],
                           forma_imagen = (448,448),
                           S = 7, B = 2):
    """
    # Argumentos:
    
        B: cantidad de Anchors(cajas contenedoras) a generar por celda
        S: cantidad de celdas a lo largo y ancho de la imagen, para crear una
            malla de SxS
        forma_imagen: tupla, con (altura, base) de la imagen para la que se crearán
            las cajas contenedoras
        escalas: lista con la dimension de 1 lado de los Anchor a calcular. Ejemplo [32,  64, 128]
        factores: lsita con los factores de tamaño de los Anchors. Ejemplo [0.5, 1 ,2]
    """
    # Obtener la combinacion de escalas y factores
    escalas, factores = numpy.meshgrid(numpy.array(escalas), numpy.array(factores))
    escalas = escalas.flatten()
    factores = factores.flatten()

    # Enumerar bases y alturas a partir de las escalas y los factores
    alturas = escalas / numpy.sqrt(factores)
    bases = escalas * numpy.sqrt(factores)

    # Enumerar centros dentro de una celda
    paso_x = (forma_imagen[0]/S)/(B+1)
    paso_y = (forma_imagen[1]/S)/(B+1)
    centros_y = numpy.arange(0, (forma_imagen[0]/S), paso_y, dtype=numpy.int8)
    centros_x = numpy.arange(0, (forma_imagen[0]/S), paso_x, dtype=numpy.int8)
    # Eliminar 0 de como posición de un centro
    centros_y = numpy.delete(centros_y,0)
    centros_x = numpy.delete(centros_x,0)
    # Invertir "y"
    centros_y=centros_y[::-1]
    
    # Inicializar variables para generar cajas
    paso_y = forma_imagen[0]/S
    paso_x = forma_imagen[1]/S
    anchors =[]
    centros =[]
    anchorsR = []
    indice_caja = -1
    for _b in range(0,B):
        for j in range(0,S):
            for i in  range(0,S):
                # Indice de caja
                indice_caja+=1
                # Espacio en pixeles, previo a la celda que se computa actualmente
                avance_y = paso_y * j
                avance_x = paso_x * i
                
                # Centros (no relativos)}
                cy = centros_y[indice_caja%len(centros_y)] + avance_y
                cx = centros_x[indice_caja%len(centros_x)] + avance_x
                # Calcular coordenadas de puntos
                y1 = int(cy - 0.5 * alturas[indice_caja%len(alturas)])
                y2 = int(cy + 0.5 * alturas[indice_caja%len(alturas)])
                x1 = int(cx - 0.5 * bases[indice_caja%len(bases)])
                x2 = int(cx + 0.5 * bases[indice_caja%len(bases)])
                
                # La caja es válida?
                # De no serlo, se corrigen sus dimensiones sin alterar
                # la posición del centro
                if y1 < 0:
                    dif = y1
                    y1 = 2
                    y2+= dif
                if x1 < 0:
                    dif = x1
                    x1 = 2
                    x2+= dif
                if y2 > forma_imagen[0]:
                    dif = y2 - forma_imagen[0]
                    y2 = forma_imagen[0]-2
                    y1+= dif
                if x2 > forma_imagen[1]:
                    dif = x2 - forma_imagen[1]
                    x2 = forma_imagen[1]-2
                    x1+= dif                
                # Añadir elementos a su respectivo arreglo
                anchors.append([y1,x1,y2,x2])
                centros.append([cy,cx])
                
    # Normalizar salidas como objeto numpy
    anchors = numpy.array(anchors,dtype=numpy.int32)
    centros = numpy.array(centros, dtype=numpy.int32)
    # Calcular valores relativos
    anchorsR = convertir_Cajas_a_Relativas(anchors,forma_imagen=forma_imagen,S=S)
    return anchors, centros, anchorsR

def generar_Anchors_Celdas_V2(escalas = [[104,  93],
                                         [ 50, 155],
                                         [ 64, 108],
                                         [ 35,  95],
                                         [144, 147],
                                         [ 34,  33],
                                         [ 14,  27],
                                         [ 85, 155],
                                         [ 69,  58]],
                              forma_imagen = (448,448),
                              S = 7, B = 2):
    """
    # Argumentos:
    
        B: cantidad de Anchors(cajas contenedoras) a generar por celda
        S: cantidad de celdas a lo largo y ancho de la imagen, para crear una
            malla de SxS
        forma_imagen: tupla, con (altura, base) de la imagen para la que se crearán
            las cajas contenedoras
        escalas: lista con la dimension de 1 lado de los Anchor a calcular. Ejemplo [32,  64, 128]
        factores: lsita con los factores de tamaño de los Anchors. Ejemplo [0.5, 1 ,2]
    """
    # Enumerar bases y alturas a partir de las escalas
    escalas  = numpy.array(escalas)
    alturas = escalas[:,1]
    bases = escalas[:,0]

    # Enumerar centros dentro de una celda
    paso_x = (forma_imagen[0]/S)/(B+1)
    paso_y = (forma_imagen[1]/S)/(B+1)
    centros_y = numpy.arange(0, (forma_imagen[0]/S), paso_y, dtype=numpy.int8)
    centros_x = numpy.arange(0, (forma_imagen[0]/S), paso_x, dtype=numpy.int8)
    # Eliminar 0 de como posición de un centro
    centros_y = numpy.delete(centros_y,0)
    centros_x = numpy.delete(centros_x,0)
    # Invertir "y"
    centros_y=centros_y[::-1]
    
    # Inicializar variables para generar cajas
    paso_y = forma_imagen[0]/S
    paso_x = forma_imagen[1]/S
    anchors =[]
    centros =[]
    anchorsR = []
    indice_caja = -1
    for _b in range(0,B):
        for j in range(0,S):
            for i in  range(0,S):
                # Indice de caja
                indice_caja+=1
                # Espacio en pixeles, previo a la celda que se computa actualmente
                avance_y = paso_y * j
                avance_x = paso_x * i
                
                # Centros (no relativos)}
                cy = centros_y[indice_caja%len(centros_y)] + avance_y
                cx = centros_x[indice_caja%len(centros_x)] + avance_x
                # Calcular coordenadas de puntos
                y1 = int(cy - 0.5 * alturas[indice_caja%len(alturas)])
                y2 = int(cy + 0.5 * alturas[indice_caja%len(alturas)])
                x1 = int(cx - 0.5 * bases[indice_caja%len(bases)])
                x2 = int(cx + 0.5 * bases[indice_caja%len(bases)])
                
                # La caja es válida?
                # De no serlo, se corrigen sus dimensiones sin alterar
                # la posición del centro
                if y1 < 0:
                    dif = y1
                    y1 = 2
                    y2+= dif
                if x1 < 0:
                    dif = x1
                    x1 = 2
                    x2+= dif
                if y2 > forma_imagen[0]:
                    dif = y2 - forma_imagen[0]
                    y2 = forma_imagen[0]-2
                    y1+= dif
                if x2 > forma_imagen[1]:
                    dif = x2 - forma_imagen[1]
                    x2 = forma_imagen[1]-2
                    x1+= dif                
                # Añadir elementos a su respectivo arreglo
                anchors.append([y1,x1,y2,x2])
                centros.append([cy,cx])
                
    # Normalizar salidas como objeto numpy
    anchors = numpy.array(anchors,dtype=numpy.int32)
    centros = numpy.array(centros, dtype=numpy.int32)
    # Calcular valores relativos
    anchorsR = convertir_Cajas_a_Relativas(anchors,forma_imagen=forma_imagen,S=S)
    return anchors, centros, anchorsR

def aplicar_Delta_Caja(caja=[],delta=[]):
    """Aplica la delta dada a la caja
    
    # Argumentos:
        
        caja: [y1, x1, y2, x2] caja a actualizar
        delta: [dy, dx, log(dh), log(dw)] refinamiento a aplicar
    
    # Salida:

        resultado: [y1, x1, y2, x2], un Anchor al que se le ha aplicado la
            transformación necesaria para ser igual a la caja contenedora real
    """
    # Convertir a y, x, h, w
    altura = caja[2] - caja[0]
    base = caja[3] - caja[1]
    centro_y = caja[0] + 0.5 * altura
    centro_x = caja[1] + 0.5 * base
    # aplicar delta
    centro_y += (delta[0] * altura)
    centro_x += (delta[1] * base)
    altura *= math.exp(delta[2])
    base *= math.exp(delta[3])
    # regresar a y1, x1, y2, x2
    y1 = centro_y - (0.5 * altura)
    x1 = centro_x - (0.5 * base)
    y2 = y1 + altura
    x2 = x1 + base
    resultado = [y1, x1, y2, x2]
    return resultado

#######################################################################################################
# Micelanea
#######################################################################################################
    