import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras as K
import keras.utils as KU
import keras.models as KM
import keras.layers as KL
import numpy

class Red:
    """ Encapsula los metodos referentes a la red proporcionados por Keras, para la construcción
    de la red neuronal convolucional para la detección de objetos.
    """
    
    def __init__(self, configuracion = None, modelo = ""):
        """
        # Argumentos:
        
            configuracion: Objeto de configuración o subclase
        
        """
        if modelo == "":
            self.config = configuracion
            tipo = self.config.RED_TIPO
            assert tipo in ['Res55', 'Conv24']
            if tipo == 'Res55':
                self.red_neuronal = self.construir_Res55(self.config)
            elif tipo == 'Conv24':
                self.red_neuronal = self.construir_Conv24(self.config)
        else:
            self.red_neuronal = KM.load_model(modelo)
        
    def construir_Res55(self, config):
        """ Genera una red con 52 capas de convolución entre las que estan implicadas algunas
        que reducen el tamaño de la matriz de imagen (Convolucional, con stride =2), en lugar
        de capaz de pooling.
        
        #Argumentos:
            
            config: objeto de la clase o subclase de Configuracion
        """
        img_entrada = self.generar_Entrada(config)
        ##############################################################################
        # Red convolucional de abstraccion de caracteristicas
        ##############################################################################

        c1 = KL.Conv2D(32, kernel_size=3, activation='relu', name="C1", padding="same")(img_entrada)

        # Rescalado (en lugar de pooling, se usa convolucional con stride = 2)
        r1 = KL.Conv2D(64, kernel_size=3, strides=2, activation='relu', name="R1", padding="same")(c1)

        # Bloque x1
        c2 = KL.Conv2D(32, kernel_size=1, activation='relu', name="C2", padding="same")(r1)
        c3 = KL.Conv2D(64, kernel_size=3, activation='relu', name="C3", padding="same")(c2)
        # Residual
        res1 = KL.add([r1,c3], name = "residual1")

        # Rescalado 2
        r2 = KL.Conv2D(128, kernel_size=3, strides=2, activation='relu', name="R2", padding="same")(res1)

        # Bloque X2
        c4 = KL.Conv2D(64, kernel_size=1, activation='relu', name="C4", padding="same")(r2)
        c5 = KL.Conv2D(128, kernel_size=3, activation='relu', name="C5", padding="same")(c4)
        #
        c6 = KL.Conv2D(64, kernel_size=1, activation='relu', name="C6", padding="same")(c5)
        c7 = KL.Conv2D(128, kernel_size=3, activation='relu', name="C7", padding="same")(c6)
        # Residual
        res2 = KL.add([r2,c7], name = "residual2")


        # Rescalado 3
        r3 = KL.Conv2D(256, kernel_size=3, strides=2, activation='relu', name="R3", padding="same")(res2)

        # Bloque X8 (1)
        c8 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C8", padding="same")(r3)
        c9 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C9", padding="same")(c8)
        #
        c10 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C10", padding="same")(c9)
        c11 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C11", padding="same")(c10)
        #
        c12 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C12", padding="same")(c11)
        c13 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C13", padding="same")(c12)
        #
        c14 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C14", padding="same")(c13)
        c15 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C15", padding="same")(c14)
        #
        c16 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C16", padding="same")(c15)
        c17 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C17", padding="same")(c16)
        #
        c18 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C18", padding="same")(c17)
        c19 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C19", padding="same")(c18)
        #
        c20 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C20", padding="same")(c19)
        c21 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C21", padding="same")(c20)
        #
        c22 = KL.Conv2D(128, kernel_size=1, activation='relu', name="C22", padding="same")(c21)
        c23 = KL.Conv2D(256, kernel_size=3, activation='relu', name="C23", padding="same")(c22)
        # Residual
        res3 = KL.add([r3,c23], name = "residual3")

        # Rescalado 4
        r4 = KL.Conv2D(512, kernel_size=3, strides=2, activation='relu', name="R4", padding="same")(res3)

        # Bloque X8 (2)
        c24 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C24", padding="same")(r4)
        c25 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C25", padding="same")(c24)
        #
        c26 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C26", padding="same")(c25)
        c27 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C27", padding="same")(c26)
        #
        c28 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C28", padding="same")(c27)
        c29 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C29", padding="same")(c28)
        #
        c30 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C30", padding="same")(c29)
        c31 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C31", padding="same")(c30)
        #
        c32 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C32", padding="same")(c31)
        c33 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C33", padding="same")(c32)
        #
        c34 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C34", padding="same")(c33)
        c35 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C35", padding="same")(c34)
        #
        c36 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C36", padding="same")(c35)
        c37 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C37", padding="same")(c36)
        #
        c38 = KL.Conv2D(256, kernel_size=1, activation='relu', name="C38", padding="same")(c37)
        c39 = KL.Conv2D(512, kernel_size=3, activation='relu', name="C39", padding="same")(c38)
        # Residual
        res4 = KL.add([r4,c39], name = "residual4")

        # Rescalado 5
        r5 = KL.Conv2D(1024, kernel_size=3, strides=2, activation='relu', name="R5", padding="same")(res4)

        # Bloque X4
        c40 = KL.Conv2D(512, kernel_size=1, activation='relu', name="C40", padding="same")(r5)
        c41 = KL.Conv2D(1024, kernel_size=3, activation='relu', name="C41", padding="same")(c40)
        #
        c42 = KL.Conv2D(512, kernel_size=1, activation='relu', name="C42", padding="same")(c41)
        c43 = KL.Conv2D(1024, kernel_size=3, activation='relu', name="C43", padding="same")(c42)
        #
        c44 = KL.Conv2D(512, kernel_size=1, activation='relu', name="C44", padding="same")(c43)
        c45 = KL.Conv2D(1024, kernel_size=3, activation='relu', name="C45", padding="same")(c44)
        #
        c46 = KL.Conv2D(512, kernel_size=1, activation='relu', name="C46", padding="same")(c45)
        c47 = KL.Conv2D(1024, kernel_size=3, activation='relu', name="C47", padding="same")(c46)
        # Residual (regresar a  c47 si falla)
        res5 = KL.add([r5,c47], name = "residual5")

        ###################################################################################
        # Red totalmente conectada (perceptrón)
        # Para clasificación y predicción de coordenadas
        ####################################################################################

        flat = KL.Flatten()(res5)
        drp = KL.Dropout(0.01)(flat)
        
        salidaT1, salidaT2 = self.generar_Salidas(drp,config)
                
        ######################################################################
        # Generacion de modelo (Red neuronal)
        ######################################################################
        modelo = KM.Model(inputs=img_entrada, outputs=[salidaT1,salidaT2])
        return modelo

    def construir_Conv24(self, config):
        """ Genera una red neuronal de 24 capas de convolución, sin shortcuts
        """
        x = self.generar_Entrada(config)
        # Bloque 1
        y = KL.Conv2D(64, kernel_size=7, activation='relu', padding="same", name="C1")(x)
        # Reescalado 1
        y = KL.MaxPool2D(pool_size=(2,2), strides=2, padding='same', name="MxP1")(y)
        # Bloque 2
        y = KL.Conv2D(192, kernel_size=3, activation='relu', padding="same", name="C2")(y)
        # Reescalado2
        y = KL.MaxPool2D(pool_size=(2,2), strides=2, padding='same', name="MxP2")(y)
        # Bloque 3
        y = KL.Conv2D(128, kernel_size=1, activation='relu', padding="same", name="C3")(y)
        y = KL.Conv2D(256, kernel_size=3, activation='relu', padding="same", name="C4")(y)
        y = KL.Conv2D(256, kernel_size=1, activation='relu', padding="same", name="C5")(y)
        y = KL.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="C6")(y)
        # Reescalado 3
        y = KL.MaxPool2D(pool_size=(2,2), strides=2, padding='same', name="MxP3")(y)
        # Bloque 4
        y = KL.Conv2D(256, kernel_size=1, activation='relu', padding="same", name="C7")(y)
        y = KL.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="C8")(y)
        y = KL.Conv2D(256, kernel_size=1, activation='relu', padding="same", name="C9")(y)
        y = KL.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="C10")(y)
        y = KL.Conv2D(256, kernel_size=1, activation='relu', padding="same", name="C11")(y)
        y = KL.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="C12")(y)
        y = KL.Conv2D(256, kernel_size=1, activation='relu', padding="same", name="C13")(y)
        y = KL.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="C14")(y)
        y = KL.Conv2D(512, kernel_size=1, activation='relu', padding="same", name="C15")(y)
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C16")(y)
        # Reescalado 4
        y = KL.MaxPool2D(pool_size=(2,2), strides=2, padding='same', name="MxP4")(y)
        # Bloque 5
        y = KL.Conv2D(512, kernel_size=1, activation='relu', padding="same", name="C17")(y)
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C28")(y)
        y = KL.Conv2D(512, kernel_size=1, activation='relu', padding="same", name="C19")(y)
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C20")(y)
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C21")(y)
        # Reescalado 5
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', strides=2, padding="same", name="C22")(y)
        # Bloque 6
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C23")(y)
        y = KL.Conv2D(1024, kernel_size=3, activation='relu', padding="same", name="C24")(y)
        # Preparar para salida
        y = KL.Flatten()(y)
        y = KL.Dropout(0.01)(y)
        
        # Generar salidas
        y1, y2 = self.generar_Salidas(y, config)
        
        modelo = KM.Model(inputs=x, outputs=[y1,y2])
        return modelo
    
    def generar_Entrada(self, config):
        """Genera la entrada de la red, segun los parametros de configuración
        """
        return KL.Input(shape=(config.FORMA_IMAGEN[0],
                                        config.FORMA_IMAGEN[1],
                                        config.FORMA_IMAGEN[2]),
                                name = "img_entrada")
    
    def generar_Salidas(self, capadrp, config):
        salida_tam  = config.S * config.S * config.B * (5 + config.NUM_CLASES)
        salida_tam2 = config.S * config.S * config.B * 5
        forma_y1 = (config.S, config.S, config.B, 5 + config.NUM_CLASES)
        forma_y2 = (config.S, config.S, config.B, 5)
        
        if config.RED_TIPO_SALIDA=="Y":
            # Tensor de salida (con formato SxSx(B*5+C))
            # Tamaño de la salida (todos los elementos sin formato)
            y1 = KL.Dense(1000, activation="sigmoid")(capadrp)
            y1 = KL.Dense(500, activation="sigmoid")(y1)
            
            y1 = KL.Dense(salida_tam, activation="sigmoid", name="salida_plana")(y1)
            y1 = KL.Reshape(forma_y1, name="tensor_salida")(y1)
            
            # Tensor  de salida para Deltas, con formato SXSX(B*5)
            # Tamaño de la salida (todos los elementos sin formato)
            y2 = KL.Dense(1000, activation="sigmoid")(capadrp)
            y2 = KL.Dense(500, activation="sigmoid")(y2)
            
            y2 = KL.Dense(salida_tam2, activation="sigmoid", name="salida_plana2")(y2)
            y2 = KL.Reshape(forma_y2, name="tensor_salida2")(y2)
            return y1,y2
        
        elif config.RED_TIPO_SALIDA=="L":
            # Tensor de salida (con formato SxSx(B*5+C))
            # Tamaño de la salida (todos los elementos sin formato)
            y1 = KL.Dense(1000, activation="sigmoid")(capadrp)
            y1 = KL.Dense(500, activation="sigmoid")(y1)
            y1 = KL.Dense(salida_tam, activation="sigmoid", name="salida_plana")(y1)
            y2 = KL.Dense(700, activation="sigmoid")(y1)
            y1 = KL.Reshape(forma_y1, name="tensor_salida")(y1)
            
            # Tensor  de salida para Deltas, con formato SXSX(B*5)
            # Tamaño de la salida (todos los elementos sin formato)
            y2 = KL.Dense(500, activation="sigmoid")(y2)
            y2 = KL.Dense(salida_tam2, activation="sigmoid", name="salida_plana2")(y2)
            y2 = KL.Reshape(forma_y2, name="tensor_salida2")(y2)
            return y1,y2
    
    
    def sumarizar_Red(self):
        """ Imprime en consola la descripción de la red neuronal que se creó, al inicializar el
        objeto. Esta descripción muestrá las conexiones que existen entre capaz y la forma de 
        su salida, asi como la cantidad de pesos y bias, y cuantos de ellos son entrenables.
        """
        self.red_neuronal.summary()
    
    def red_a_IMG(self, archivo="miModelo.png"):
        """ Guarda un esquema de la red creada en un archivo de imagen
        
        #Argumentos:
            archivo: nombre del archivo, indicando el formato .png, la dirección relativa
                ejemplo: miRed.png
        """
        KU.plot_model(self.red_neuronal, to_file=archivo)
