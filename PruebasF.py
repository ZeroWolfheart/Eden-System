import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from Configuracion import Configuracion
from Red import Red
from Dataset import Dataset
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import math

import Utiles
import Modelo
import keras.optimizers as KO
import keras.callbacks as KC

miConfig = Configuracion()
#miRed = Red(configuracion=miConfig)
miRed = Red(modelo="pesos/Eden_SystemV3_kagaroo_v2_0098.h5")
miRed.sumarizar_Red()
miRed.red_a_IMG()


miData = Dataset("kagaroo_v2",  "kangaroo-master")
miData.agregar_Clase("kangaroo")

#miData.agregar_Clase("aeroplane")
#miData.agregar_Clase("bicycle")
#miData.agregar_Clase("bird")
#miData.agregar_Clase("boat")
#miData.agregar_Clase("bottle")
#miData.agregar_Clase("bus")
#miData.agregar_Clase("car")
#miData.agregar_Clase("cat")
#miData.agregar_Clase("chair")
#miData.agregar_Clase("cow")
#miData.agregar_Clase("diningtable")
#miData.agregar_Clase("dog")
#miData.agregar_Clase("horse")
#miData.agregar_Clase("motorbike")
#miData.agregar_Clase("person")
#miData.agregar_Clase("pottedplant")
#miData.agregar_Clase("sheep")
#miData.agregar_Clase("sofa")
#miData.agregar_Clase("train")
#miData.agregar_Clase("tvmonitor")

miData.cargar_Dataset()
miData.crear_SubSets()

sgd = KO.SGD(lr=0.0001, momentum=0.9, decay=0.0001)
miRed.red_neuronal.compile(
    optimizer = sgd,
    loss = 'mean_squared_error'
)

train_gen = Modelo.generador_Datos(dataset=miData,
                                   modo="entrenamiento",
                                   configuracion=miConfig,
                                   revolver=True,
                                   tam_batch=miConfig.TAM_BATCH)
val_gen = Modelo.generador_Datos(dataset=miData,
                                   modo="validacion",
                                   configuracion=miConfig,
                                   revolver=True,
                                   tam_batch=miConfig.TAM_BATCH)

pasos = math.ceil(len(miData.entrenamiento) / miConfig.TAM_BATCH)
val_pasos = math.ceil(len(miData.validacion) / miConfig.TAM_BATCH)

checkpoint_path = os.path.join("pesos", "Eden_SystemV3_{}_*epoch*.h5".format(miData.nombre_Dataset))
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

cvs_path = os.path.join("logs", "training_{}.log".format(miData.nombre_Dataset))

# Create log_dir if it does not exist
if not os.path.exists("pesos"):
    os.makedirs("pesos")
# Create log_dir if it does not exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Callbacks
callbacks = [
    KC.CSVLogger(cvs_path, separator=',',  append=True),
    KC.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
]

miRed.red_neuronal.fit_generator(
            train_gen,
            initial_epoch=140,
            epochs=200,
            steps_per_epoch=pasos,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_pasos,
            max_queue_size=10,
            workers=4,
            use_multiprocessing=True
        )
