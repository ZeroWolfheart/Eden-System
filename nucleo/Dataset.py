from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import skimage.color
import skimage.io
import skimage.transform

class Dataset:
    
    def __init__(self, nombre, direccion):
        self.nombre_Dataset = nombre
        self.direccion_Raiz = direccion #dirección que contiene las carpetas de imagenes y anotaciones
        self.clases = []                # nombres de las clases que contiene el Dataset
        self.informacion_Imagenes = []
        self.entrenamiento = []         # informaición del conjunto de entrenamiento
        self.validacion = []            # informaición del conjunto de validación

    def agregar_Clase(self, nombre_Clase):
        """ Añade 1 elemento al conjunto de clases que posee el Dataset.
        
        # Argumentos:
        
            nombreClase: cadena con el nombre de la clase;  debe ser identico a como esta escrito en el archivo de anotaciones
        """
        if (self.clases.count(nombre_Clase) == 0):
            self.clases.append(nombre_Clase)
        else:
            print('Nombre de clase repetido')

    # obtener direccion de imagen
    def getDireccionImagen(self, idImagen, conjunto):
        info = conjunto[idImagen]
        return info['rI']

    def cargar_Dataset(self):
        """ Estrae y lista los archivos de imagen y sus respectivas anotaciones de la
        dirección donde se aloja el dataset y lo guarda  en informacion_Imagenes
        con formato [{'id':nombre imagen sin extencion, 'rI': ruta de la imagen, 'rA':  ruta de anotaciones},...]
        """
        # definir directorios de datos
        dir_Imagenes     = self.direccion_Raiz + '/images/'
        dir_Anotaciones  = self.direccion_Raiz + '/annots/'

        # encontrar todas las imagenes
        for nombre_Archivo in listdir(dir_Imagenes):
            # extraer id de imagen
            punto = nombre_Archivo.find(".")
            extencion = len(nombre_Archivo)-punto
            id_Imagen = nombre_Archivo[:-extencion]
            # construir rutas de imagen y anotaciones
            ruta_Imagen = dir_Imagenes + nombre_Archivo
            ruta_Anotacion = dir_Anotaciones + id_Imagen + '.xml'
            # añadir datos a la lista (en otra lista, con formato id de imagen, ruta de imagne y ruta de anotaciones)
            self.informacion_Imagenes.append({'id': id_Imagen,'rI':ruta_Imagen, 'rA':ruta_Anotacion})

    def extraer_Cajas(self, ruta_Archivo):
        """ Lee la información de un archivo ".xml" de anotaciones, y obtiene de ella
        las cajas contenedoras que rodean a los objetos dentro de la imagen así 
        como la clase a la que pertence. 
        
        # Argumentos:
        
            ruta_Archivo: cadena con la ruta y nombre del archivo
        
        # Salidas:
        
            lista cajas contenedoras: formato [[xmin, ymin, xmax, ymax, nombre],...]
                donde: 
                    xmin, ymin, xmax, ymax: son las coordenadas en pixeles de
                        las esquinas que delimitan la caja
                    nombre: es una cadena, con el nombre de la clase, tal cual
                        aparece en el archivo examindado
                        
            base: de la imagen en pixeles

            altura: de la imagen en pixeles
        """
        # cargar y analizar el archivo
        arbol = ElementTree.parse(ruta_Archivo)
        # obtener raíz del documento
        raiz = arbol.getroot()
        # extraer cada caja contenedora
        cajas = list()
        for objeto in raiz.findall('./object'):
            nombre = objeto.find('name').text
            box = objeto.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax, nombre]
            cajas.append(coors)
        # extraer dimensiones de imagen
        base = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return cajas, base, altura
 
    def cargar_Mascara(self, indice, conjunto):
        """  Utiliza el método extraer_Cajas(), y con base en la información obtenida de
        cada caja contenedora, la base y altura de la imagen, genera un objeto numpy,
        con la máscara de la imagen  y los objetos que en ella se encuentran identificados,
        cada caja, enmascarada en un canal distinto.
        
        # Argumentos: 

            indice: entero, que indica la posicición en el conjunto del elemento a cargar
                dentro del conjunto dado.
            conjunto: lista, en formato[{'id':,'rI':, 'rA': },...], puede ser informacion_Imagenes,
                validacion o entrenamiento.
        
        # Salida:
        
            objeto numpy: de forma (altura, base, cantidad de cajas) que contiene la información
            de la posición de los objetos en la imagen original, según las anotaciones
            
            arreglo: de longitud = cantidad  de cajas, que contienen el id (indice) de la clase
            a la que  pertenece la caja, segun el arreglo clases.
        """
        # obtener detalles de imagen
        info = conjunto[indice]
        # obtener direccion de anotaciones
        ruta = info['rA']
        # cargar XML
        cajas, w, h = self.extraer_Cajas(ruta)
        # crear un arreglo para todas las mascaras, cada una en un canal diferente
        mascaras = zeros([h, w, len(cajas)], dtype='uint8')
        # crear mascaras
        class_ids = list()
        for i in range(len(cajas)):
            box = cajas[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            mascaras[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.clases.index(box[4]))
        return mascaras, asarray(class_ids, dtype='int32')

    def cargar_Imagen(self, indice, conjunto):
        """ Lee un archivo de imagen y lo codifica en un arreglo de matrices.
        Una matriz por cada canal RGB. Si la imagen posee  una cantidad distinta
        de canales, esta se  normaliza y se devuelve en tres canales.
        
        # Argumentos: 

            indice: entero, que indica la posicición en el conjunto del elemento a cargar
                dentro del conjunto dado.
            conjunto: lista, en formato[{'id':,'rI':, 'rA': },...], puede ser informacion_Imagenes,
                validacion o entrenamiento.
        
        # Salida:
        
            objeto numpy, de forma (altura, base, 3) que contiene la información de la imagen
            original separa en los canales RGB
        """
        # obtener detalles de imagen
        info = conjunto[indice]
        # obtener direccion de anotaciones
        ruta = info['rI']
        # cargar imagen
        imagen = skimage.io.imread(ruta)
        # Si esta en escala de grises, convertir en RGB para mantener consistencia
        if imagen.ndim != 3:
            imagen = skimage.color.gray2rgb(imagen)
        # Sí tiene un canal Alpha, remover para mantener consistencia
        if imagen.shape[-1] == 4:
            imagen = imagen[..., :3]
        return imagen
    
    def crear_SubSets(self, porcentaje = 0.8):
        """ Crea los conjuntos de validación y entrenamiento.
        Cuantifica la cantidad de imagenes que existen para cada clase
        Separa el porcentaje especificado de ellas para entrenamiento  y el resto las
        coloca en el conjunto de validacion.
            
            Este metodo establece las listas entrenamiento y validación, con elementos que
        pertenecen a informacion_Imagenes, en el mismo formato que este.
        
        # Argumentos:
        
            porcentaje: decimal, menor a 1, representa el procentaje de imagenes del dataset
                que se utilizarán para el entrenmaiento, el resto se utilizaran para validación
                
        # Salidas:
            
            None
        """
        # Iniciar lista que contendra por separado las imagenes de cada clase
        imgClase = list(range(len(self.clases)))
        for j in range(len(imgClase)):
            imgClase[j] = list()
        # Verificar cada imagen, para saber a que clase pertenece
        # Sí una imagen contiene más de una clase se incluira en todas aquellas que tenga
        for elemento in range(len(self.informacion_Imagenes)):
            mascaras, idsClase = self.cargar_Mascara(elemento, self.informacion_Imagenes)
            del mascaras # eliminar variable que no se usa para ahorrar espacio en memoria
            # Identificar clases en una imagen
            for id in range(len(self.clases)):
                # Verificar si existe la clase en el arreglo de clases
                if(idsClase.tolist().count(id)>0):
                    # Añadir elemento al arreglo de imagenes separadas por clase
                    imgClase[id].append(elemento)   
        # Cuantificar imagenes por clase y seleccionar cantidad para cada set
        cantImgClase = list(range(len(imgClase))) #variable donde se guarda cantidad por clase
        for i in range(len(imgClase)):
            tot = len(imgClase[i])
            ent = int(tot* porcentaje)
            val = tot - ent
            cantImgClase[i] = {'total':tot, 'entrenar': ent, 'validar': val}
            # print('Imagenes de clase ', i ,' : ',cantImgClase[i])
        # Separar las imagenes en los conjuntos de validación y entrenamiento
        # segun la cantidad indicada para cada clase
        for i in range(len(imgClase)):
            ent =  cantImgClase[i]['entrenar']
            contador = 0
            for j in range (len(imgClase[i])):
                if (contador < ent):
                    self.entrenamiento.append(self.informacion_Imagenes[imgClase[i][j]])
                else:
                    self.validacion.append(self.informacion_Imagenes[imgClase[i][j]])
                contador+=1
        return None