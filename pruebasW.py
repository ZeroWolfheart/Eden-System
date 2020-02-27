import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras as K
import keras.utils as KU
import keras.models as KM
import keras.layers as KL
import numpy

# [[b,h]...]
wh = numpy.array(
    [[1,1],
    [1,2],
    [1,3],
    [1,4],
    [1,5],
    [1,6],
    [1,7],
    [1,8],
    [1,9],
    [1,10],
    [2,1],
    [2,2],
    [2,3],
    [2,4],
    [2,5],
    [2,6],
    [2,7],
    [2,8],
    [2,9],
    [2,10],
    [3,1],
    [3,2],
    [3,3],
    [3,4],
    [3,5],
    [3,6],
    [3,7],
    [3,8],
    [3,9],
    [3,10],
    [4,1],
    [4,2],
    [4,3],
    [4,4],
    [4,5],
    [4,6],
    [4,7],
    [4,8],
    [4,9],
    [4,10],
    [4,1],
    [4,2],
    [4,3],
    [4,4],
    [4,5],
    [4,6],
    [4,7],
    [4,8],
    [4,9],
    [4,10],
    [5,1],
    [5,2],
    [5,3],
    [5,4],
    [5,5],
    [5,6],
    [5,7],
    [5,8],
    [5,9],
    [5,10],
    [6,1],
    [6,2],
    [6,3],
    [6,4],
    [6,5],
    [6,6],
    [6,7],
    [6,8],
    [6,9],
    [6,10],
    [7,1],
    [7,2],
    [7,3],
    [7,4],
    [7,5],
    [7,6],
    [7,7],
    [7,8],
    [7,9],
    [7,10],
    [8,1],
    [8,2],
    [8,3],
    [8,4],
    [8,5],
    [8,6],
    [8,7],
    [8,8],
    [8,9],
    [8,10],
    [9,1],
    [9,2],
    [9,3],
    [9,4],
    [9,5],
    [9,6],
    [9,7],
    [9,8],
    [9,9],
    [9,10],
    [10,1],
    [10,2],
    [10,3],
    [10,4],
    [10,5],
    [10,6],
    [10,7],
    [10,8],
    [10,9],
    [10,10],
    [9.6, 2.5],
    [3.4,1.2],
    [4.5,5],
    [6.3,7.2],
    [5.5,6.6],
    [6.4,8.9],
    [2.5,3.6],
    [9.8,3.2],
    [2.3,4.4],
    [5.6,1.6],
    [7.9,6.8]],
    dtype=numpy.float
)

area = numpy.array(
    [[0.5],
    [1],
    [1.5],
    [2],
    [2.5],
    [3],
    [3.5],
    [4],
    [4.5],
    [5],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [1.5],
    [3],
    [4.5],
    [6],
    [7.5],
    [9],
    [10.5],
    [12],
    [13.5],
    [15],
    [2],
    [4],
    [6],
    [8],
    [10],
    [12],
    [14],
    [16],
    [18],
    [20],
    [2],
    [4],
    [6],
    [8],
    [10],
    [12],
    [14],
    [16],
    [18],
    [20],
    [2.5],
    [5],
    [7.5],
    [10],
    [12.5],
    [15],
    [17.5],
    [20],
    [22.5],
    [25],
    [3],
    [6],
    [9],
    [12],
    [15],
    [18],
    [21],
    [24],
    [27],
    [30],
    [3.5],
    [7],
    [10.5],
    [14],
    [17.5],
    [21],
    [24.5],
    [28],
    [31.5],
    [35],
    [4],
    [8],
    [12],
    [16],
    [20],
    [24],
    [28],
    [32],
    [36],
    [40],
    [4.5],
    [9],
    [13.5],
    [18],
    [22.5],
    [27],
    [31.5],
    [36],
    [40.5],
    [45],
    [5],
    [10],
    [15],
    [20],
    [25],
    [30],
    [35],
    [40],
    [45],
    [50],
    [12],
    [4.08],
    [11.25],
    [22.68],
    [18.15],
    [(6.4*8.9)/2],
    [(2.5*3.6)/2],
    [(9.8*3.2)/2],
    [(2.3*4.4)/2],
    [(5.6*1.6)/2],
    [(7.9*6.8)/2]],
    dtype = numpy.float
)
whT = numpy.array(
    [[6.4, 8.2],
    [1.2, 6.5],
    [9.0, 4.4],
    [7.6, 10],
    [6.3, 4.0],
    [0.8, 2.0],
    [3.3, 3.3],
    [5.0, 2.6],
    [9.1, 5.5],
    [9.0, 4.6],
    [3.9, 5.8],
    [4.3, 9.6],
    [2.3, 4.4],
    [6.8, 1.2],
    [7.7, 6.0],
    [6.6, 8.0]])
aT = numpy.array(
    [[(6.4* 8.2)/2],
    [(1.2* 6.5)/2],
    [(9.0* 4.4)/2],
    [(7.6* 10)/2],
    [(6.3* 4.0)/2],
    [(0.8* 2.0)/2],
    [(3.3* 3.3)/2],
    [(5.0* 2.6)/2],
    [(9.1* 5.5)/2],
    [(9.0* 4.6)/2],
    [(3.9* 5.8)/2],
    [(4.3* 9.6)/2],
    [(2.3* 4.4)/2],
    [(6.8* 1.2)/2],
    [(7.7* 6.0)/2],
    [(6.6* 8.0)/2]]
)

# Escalar, para que ningun valor sea mayor de 1
wh1 = wh/10
area1 = area/100
whT1 = whT/10
aT1 = aT/100


"""
#Modelo secuencial

modelo = KM.Sequential()
modelo.add(KL.Dense(6, input_shape=(2,)))
modelo.add(KL.Dense(6, activation="sigmoid"))
modelo.add(KL.Dense(15, activation="sigmoid"))
modelo.add(KL.Dense(30, activation="sigmoid"))
modelo.add(KL.Dense(10, activation="sigmoid"))
modelo.add(KL.Dense(5, activation="sigmoid"))
modelo.add(KL.Dense(1, activation="sigmoid"))
modelo.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)
#100000
#modelo.fit(wh1, area1, epochs=100000, batch_size=32)
#puntuacion = modelo.evaluate(whT1, aT1, batch_size=16)

#modelo.save("barba1")
#modelo.save_weights("barba1_weights")
modelo.summary()
KU.plot_model(modelo, to_file="estacosa.png")
"""

entrada = KL.Input(shape=(2,))
capa1 = KL.Dense(6,  activation="sigmoid")(entrada)
capa2 = KL.Dense(6,  activation="sigmoid")(capa1)
capa3 = KL.Dense(15, activation="sigmoid")(capa2)
capa4 = KL.Dense(30, activation="sigmoid")(capa3)
capa5 = KL.Dense(10, activation="sigmoid")(capa4)
capa6 = KL.Dense(5,  activation="sigmoid")(capa5)
salida = KL.Dense(1, activation="sigmoid")(capa6)

modelo2 = KM.Model(inputs = entrada, outputs = salida)
modelo2.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

#100000
modelo2.fit(wh1, area1, epochs=100000, batch_size=32)
puntuacion = modelo2.evaluate(whT1, aT1, batch_size=16)
print(puntuacion)

modelo2.save("barba2")
modelo2.save_weights("barba2_weights")

#modelo2.summary()
#KU.plot_model(modelo2, to_file="estacosa2.png")
