import random as rn
import numpy
from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle

from Configuracion import Configuracion
import Utiles

def imprimir_Anchors(anchors = None, img = None, config = Configuracion):
    _, ax = pyplot.subplots(1, figsize=(16,16))
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("Anchors")

    for anchor in anchors:
        
        y1, x1, y2, x2 = anchor
        r,g,b = rn.random(), rn.random(), rn.random()
        amr = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1.5, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr)
        amr2 = Circle((x1+(0.5*(x2-x1)) , y1+(0.5*(y2-y1))), radius=0.5, linewidth=1.5, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr2)
            
    pyplot.imshow(img)
    pyplot.show()

def imprimir_Predicciones(imagen = None, configuracion = Configuracion,
                          anchors_Propuestos = None, deltas_Calculados = None,
                          clases_Anchor = None, clases = []):
    # Imprimir imagen (ventana contenedora)
    _, ax = pyplot.subplots(1, figsize=(16,16))
    altura, base = imagen.shape[:2]
    ax.set_ylim(altura + 10, -10)
    ax.set_xlim(-10, base + 10)
    ax.axis('off')
    ax.set_title("Resultado")

    # Imprimir anchors_propuestos, deltas_calculados, clases_anchor
    # Para prediccion
    for i in range(0,len(anchors_Propuestos)):
        # Valores de la caja (tensor 1)
        y1, x1, y2, x2, iou = anchors_Propuestos[i]
        # Si la red tiene dos tensores
        if configuracion.RED_TIPO_SALIDA in ['Y','L']:
            # Si esta activado el identificador de anchor positivo
            if configuracion.USAR_IDF:
                dy,dx,logdh,logdw,idf = deltas_Calculados[i]
                #TODO: Expiremental...
                iou=idf
            else:
                dy,dx,logdh,logdw = deltas_Calculados[i]
            # Funcion que unifica los valores devueltos por ambos
            # tensores en las coordenadas de la caja
            y1,x1,y2,x2=Utiles.aplicar_Delta_Caja(caja=[y1,x1,y2,x2],delta=[dy,dx,logdh,logdw])

        # Umbral de detección
        if iou > configuracion.DETECCION_CONFIDENCIA_MINIMA:
            # Generar color de la caja
            r,g,b = rn.random(), rn.random(), rn.random()
            # Selccionar clase a la que pertenece el elemento
            cc = numpy.argmax(clases_Anchor[i])
            
            #TODO: Comentar estas 3 lineas
            print([y1,x1,y2,x2])
            print(clases_Anchor[i])
            print(clases[cc])
            # Crear caja
            box = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
            ax.add_patch(box)
            # Formato de % de confidencia
            porc  = "{0:.2f}".format(iou*100)
            # Crear etiqueta de la caja
            ax.text(x1+2,y1 + 8, "{}%: {}".format(porc, clases[cc]), color=(r,g,b), size=11, backgroundcolor="none")
    
    # Crear e imprimir imagen en el contenedor
    pyplot.imshow(imagen)
    pyplot.show()

def dibujar_Malla(configuracion = Configuracion, imagen = None):
    # Dibujar malla en imagen
    if configuracion.RESP_MALLA:
        mx,my = imagen.shape[0]//configuracion.S, imagen.shape[1]//configuracion.S
        imagen[:,::my,:] = configuracion.RESP_COLOR_MALLA
        imagen[::mx,:,:] = configuracion.RESP_COLOR_MALLA
    return imagen