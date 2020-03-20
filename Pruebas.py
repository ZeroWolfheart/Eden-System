from Configuracion import Configuracion
from Red import Red
from Dataset import Dataset
import Modelo
import Utiles
import Visual

from matplotlib import pyplot
from matplotlib.patches import Rectangle

import numpy
import random as rn
import math

miConfig = Configuracion()
miRed = Red(configuracion=miConfig)
miRed.sumarizar_Red()

miData = Dataset("Frutero",  "frutas")
miData.agregar_Clase("apple")
miData.agregar_Clase("banana")
miData.agregar_Clase("orange")

# miData.agregar_Clase("car")
# miData.agregar_Clase("person")
# miData.agregar_Clase("aeroplane")
# miData.agregar_Clase("bicycle")
# miData.agregar_Clase("bird")
# miData.agregar_Clase("boat")
# miData.agregar_Clase("bottle")
# miData.agregar_Clase("bus")
# miData.agregar_Clase("cat")
# miData.agregar_Clase("chair")
# miData.agregar_Clase("cow")
# miData.agregar_Clase("diningtable")
# miData.agregar_Clase("dog")
# miData.agregar_Clase("horse")
# miData.agregar_Clase("motorbike")
# miData.agregar_Clase("pottedplant")
# miData.agregar_Clase("sheep")
# miData.agregar_Clase("sofa")
# miData.agregar_Clase("train")
# miData.agregar_Clase("tvmonitor")

miData.cargar_Dataset()
miData.crear_SubSets()


# miImagen = 22
miImagen = 88
# Original
pic = miData.cargar_Imagen(miImagen, miData.entrenamiento)
mask, ids_clase = miData.cargar_Mascara(miImagen, miData.entrenamiento)

# Reescalar
pic, ventana, escala, relleno, aleatorio = Utiles.reescalar_Imagen(pic, minDim=miConfig.MIN_DIM, maxDim=miConfig.MAX_DIM,
                              minEscala=miConfig.ESCALA_MINIMA, modo="cuadrado")
mask = Utiles.reescalar_Mascara(mask,escala,relleno,aleatorio=aleatorio)
cajas = Utiles.extraer_Cajas_Contenedoras(mask)

anclas, centrosA, anclasR = Utiles.generar_Anchors_Celdas_V2(miConfig.ANCHOR_SCALAS,forma_imagen=(miConfig.FORMA_IMAGEN[0],miConfig.FORMA_IMAGEN[1]), S=miConfig.S, B=miConfig.B)

Visual.imprimir_Anchors(anchors=anclas,img=pic,config=miConfig)

iou = Utiles.calcular_Sobreposiciones(anclas, cajas)
# Calcular desviación entre caja contenedora y ancla
identificador, cajasDelta, mejorCoincidencia = Modelo.calcular_Deltas(anclas,
                                                            cajas,
                                                            iou,
                                                            miConfig)
if miConfig.RED_TIPO_SALIDA in ["L","Y"]:
    
    
    t1, t2 = Modelo.codificar_tensores_Salida(S=miConfig.S,
                                    B=miConfig.B,
                                    C=miConfig.NUM_CLASES,
                                    anchors=anclasR,
                                    iou=iou,
                                    ids_Clase=ids_clase,
                                    deltas = cajasDelta,
                                    identificador = identificador,
                                    mejor_Coincidencia=mejorCoincidencia,
                                    usar_Idf=miConfig.USAR_IDF)
    
elif miConfig.RED_TIPO_SALIDA in ["I"]:
    # Generar ubicación relativa de cajas contenedoras
    cajasR = Utiles.convertir_Cajas_a_Relativas(cajas,
                                                forma_imagen=(miConfig.FORMA_IMAGEN[0],miConfig.FORMA_IMAGEN[1]),
                                                S = miConfig.S)

    t1 = Modelo.codificar_Unico_Tensor_Salida(S=miConfig.S,
                                            B=miConfig.B,
                                            C=miConfig.NUM_CLASES,
                                            anchors=anclasR,
                                            cajasR= cajasR,
                                            ids_Clase=ids_clase,
                                            identificador = identificador,
                                            deltas=cajasDelta,
                                            mejor_Coincidencia=mejorCoincidencia,
                                            IoU = iou)
    #print(t1)

if miConfig.RED_TIPO_SALIDA in ["L","Y"]:
    anchors_propuestos, deltas_calculados, clases_anchor =  Modelo.decodificar_Tensores(t1=t1,
                                                                                    t2=t2,
                                                                                    forma_Imagen=(miConfig.FORMA_IMAGEN[0], miConfig.FORMA_IMAGEN[1]),
                                                                                    S=miConfig.S,
                                                                                    B=miConfig.B,
                                                                                    usar_Idf=miConfig.USAR_IDF)
elif miConfig.RED_TIPO_SALIDA in ["I"]:
    anchors_propuestos, clases_anchor = Modelo.decodificar_Unico_Tensor_Salida( t1=t1,
                                                                                anchors=anclas,
                                                                                S=miConfig.S,
                                                                                B=miConfig.B,
                                                                                forma_imagen=(miConfig.FORMA_IMAGEN[0],miConfig.FORMA_IMAGEN[1]))
# Forma de imprimir
mx,my = pic.shape[0]//miConfig.S, pic.shape[1]//miConfig.S
color_malla=[255,255,255]
pic[:,::my,:] = color_malla
pic[::mx,:,:] = color_malla

_, ax = pyplot.subplots(1, figsize=(16,16))
height, width = pic.shape[:2]
ax.set_ylim(height + 10, -10)
ax.set_xlim(-10, width + 10)
ax.axis('off')
ax.set_title("la propuesta")

#imprimir anchors_propuestos, deltas_calculados, clases_anchor
for i in range(0,len(anchors_propuestos)):
    y1, x1, y2, x2, iou = anchors_propuestos[i]
    if miConfig.RED_TIPO_SALIDA in ["L","Y"]:
        if miConfig.USAR_IDF:
            dy,dx,logdh,logdw,idf = deltas_calculados[i]
        else:
            dy,dx,logdh,logdw = deltas_calculados[i]
    
    r,g,b = rn.random(), rn.random(), rn.random()
    cc = numpy.argmax(clases_anchor[i])
    
    # indicador
    if iou > miConfig.DELTA_IOU_MIN_POSITIVO:
    #if idf == 1:
        zz=3
        if miConfig.RED_TIPO_SALIDA in ["L","Y"]:
            print(dy,dx,logdh,logdw)
            deltx =  Utiles.aplicar_Delta_Caja(caja=[y1,x1,y2,x2],delta=[dy,dx,logdh,logdw])
            print(deltx)
            amr2 = Rectangle((deltx[1],deltx[0]), deltx[3]-deltx[1], deltx[2]-deltx[0], linewidth=zz, alpha=0.7, linestyle='dashed', edgecolor=(r,g,b), facecolor='none')
            ax.add_patch(amr2)
        
        # print([y1,x1,y2,x2])
        # print(cc, iou)
        
        #ax.text(deltx[1]+2,deltx[0] + 8, "{}%: {}".format(iou*100, miData.clases[cc]), color='w', size=11, backgroundcolor="none")
        ax.text(x1+2 ,y1+8, "{}%: {}".format(iou*100, miData.clases[cc]), color='w', size=11, backgroundcolor="none")
        
        amr = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
        ax.add_patch(amr)
        
    # amr = Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1, alpha=0.7, linestyle='solid', edgecolor=(r,g,b), facecolor='none')
    # ax.add_patch(amr)

pyplot.imshow(pic)
for i in range(mask.shape[2]):
    pyplot.imshow(mask[:,:,i], cmap='gray', alpha=0.2)

pyplot.show()

