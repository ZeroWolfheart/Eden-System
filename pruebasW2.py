import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras as K
import keras.utils as KU
import keras.models as KM
import keras.layers as KL
import numpy


modelo = KM.load_model("barba2")
modelo.load_weights("barba2_weights")

test = numpy.array([[10,10]])
test1 = test/10

prediction = modelo.predict(test1)

print(prediction)
print(prediction*100)