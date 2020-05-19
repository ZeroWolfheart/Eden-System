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
miRed = Red(configuracion=miConfig)
#miRed = Red(modelo="pesos/Eden_SystemV4_Frutero_0395.h5")
miRed.sumarizar_Red()
miRed.red_a_IMG()
miRed.red_neuronal.load_weights("pesos/frutas/Eden_SystemV4_Frutero_13751.h5")

miData = Dataset("Frutero",  "datasets/frutas")
miData.agregar_Clase("apple")
miData.agregar_Clase("banana")
miData.agregar_Clase("orange")

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

sgd = KO.SGD(lr=0.001,
             momentum=0.5,
             decay=0.0001)
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

checkpoint_path = os.path.join("pesos", "Eden_SystemV4_{}_*epoch*.h5".format(miData.nombre_Dataset))
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

cvs_path = os.path.join("logs", "training_{}.log".format(miData.nombre_Dataset))

# Create log_dir if it does not exist
if not os.path.exists("pesos/frutas/"):
    os.makedirs("pesos/frutas/")
# Create log_dir if it does not exist
if not os.path.exists("logs"):
    os.makedirs("logs")

auto_borrador = KC.LambdaCallback(
    on_epoch_end=lambda epoch, logs: borrador("pesos/")

)
def borrador (direcccion):
    saves = os.listdir(direcccion)
    if len(saves) > 3:
        saves = sorted(saves, reverse=True)
        borrables = saves[3:]
        for borrable in borrables:
            os.remove(direcccion+borrable)
        

# Callbacks
callbacks = [
    KC.CSVLogger(cvs_path, separator=',',  append=True),
    KC.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False),
    auto_borrador
]
print(callbacks)
miRed.red_neuronal.fit_generator(
            train_gen,
            initial_epoch=20,
            epochs=5,
            steps_per_epoch=pasos,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_pasos,
            max_queue_size=10,
            #workers=2,
            use_multiprocessing=False,
            verbose=2
        )
# miRed.red_neuronal.save("pesos/epoca_{}".format(epoca), overwrite=True, include_optimizer=True)
